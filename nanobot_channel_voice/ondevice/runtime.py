"""A minimal RKNN / ONNX model runtime, shared by the on-device adapters, mirroring
the ``init_model`` / ``run_*`` / ``release_model`` pattern of the Rockchip
rknn_model_zoo demos the adapters are ported from. Backend follows the file
extension: ``.rknn`` on the Rockchip NPU, ``.onnx`` via ``onnxruntime``, and those
two are the only supported artifacts.

``.onnx`` execution providers are configurable, so the SAME export runs on CPU
(default) or an accelerator: on NVIDIA Jetson pass ``["TensorrtExecutionProvider",
"CUDAExecutionProvider", "CPUExecutionProvider"]`` (the TensorRT EP builds and
caches an engine from that same ``.onnx``, no separate artifact), always keeping
``CPUExecutionProvider`` last as the fallback for unsupported nodes;
``provider_options`` is the parallel per-provider option list.

Heavy runtimes import lazily; a construction failure is raised to the registry,
which warns and falls back to a cloud/system backend.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from loguru import logger

# (name, array) pairs: ONNX is fed by name, RKNN positionally, so this carries both.
NamedInputs = list[tuple[str, Any]]


def _core_mask(api: Any, core_mask: str) -> Any:
    """Resolve ``core_mask`` to the runtime's ``NPU_CORE_*`` constant (RK3588 has
    3 cores; AUTO lets the runtime decide). Older toolkits lack some names: say
    so rather than silently running on one core."""
    name = f"NPU_CORE_{core_mask.upper()}"
    mask = getattr(api, name, None)
    if mask is None:  # NPU_CORE_AUTO == 0, so this must test None, not falsiness
        logger.warning("voice: this RKNN runtime has no {}; using NPU_CORE_AUTO", name)
        return api.NPU_CORE_AUTO
    return mask


def _load_rknn(path: str, *, core_mask: str, target: str | None, device_id: str | None) -> Any:
    """Load a ``.rknn`` model, preferring ``rknn-toolkit-lite2`` (the on-board runtime;
    the ``[rknn]`` extra, aarch64-only wheels) with fallback to the full
    ``rknn-toolkit2`` (an x86 box with only the converter); the toolkits differ only in
    ``init_runtime``, where the full one additionally takes ``target=``/``device_id=``."""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        pass
    else:
        rknn = RKNNLite()
        if rknn.load_rknn(path) != 0:
            rknn.release()
            raise RuntimeError(f"failed to load RKNN model: {path}")
        if rknn.init_runtime(core_mask=_core_mask(RKNNLite, core_mask)) != 0:
            rknn.release()
            raise RuntimeError(f"failed to init RKNNLite runtime for: {path}")
        # Drop lite2's copy of the whole model file — dead weight once init_runtime has
        # loaded the NPU (487 MB for SenseVoice); getattr: absent on the full toolkit.
        if getattr(rknn, "rknn_data", None) is not None:
            rknn.rknn_data = None
        return rknn

    from rknn.api import RKNN

    rknn = RKNN()
    if rknn.load_rknn(path) != 0:
        rknn.release()
        raise RuntimeError(f"failed to load RKNN model: {path}")
    if rknn.init_runtime(
        target=target, device_id=device_id, core_mask=_core_mask(RKNN, core_mask)
    ) != 0:
        rknn.release()
        raise RuntimeError(f"failed to init RKNN runtime for: {path}")
    return rknn


class OnDeviceModel:
    """A single loaded model (one encoder or one decoder)."""

    def __init__(
        self,
        path: str,
        *,
        core_mask: str = "auto",
        target: str | None = None,
        device_id: str | None = None,
        providers: list | None = None,
        provider_options: list | None = None,
        intra_op_threads: int | None = None,
        rknn_input_permutation: tuple[int, ...] | None = None,
    ):
        self._rknn: Any = None
        self._sess: Any = None
        self._released = False
        self._rknn_lock = threading.Lock()  # RKNN contexts are not thread-safe
        # Pre-transpose RKNN inputs of matching rank into the import's
        # reported layout (an import can fold a layout swap into the graph);
        # declaring data_format per call instead makes Lite repair-and-WARN
        # at every inference.
        if rknn_input_permutation is not None:
            if sorted(rknn_input_permutation) != list(range(len(rknn_input_permutation))):
                raise ValueError("rknn_input_permutation must be a permutation of input axes")
            import numpy

            self._np = numpy
        self._rknn_input_permutation = rknn_input_permutation

        if path.endswith(".rknn"):
            self._rknn = _load_rknn(path, core_mask=core_mask, target=target, device_id=device_id)
        elif path.endswith(".onnx"):
            import onnxruntime

            kwargs: dict[str, Any] = {"providers": providers or ["CPUExecutionProvider"]}
            if provider_options is not None:
                kwargs["provider_options"] = provider_options
            opts = onnxruntime.SessionOptions()
            if intra_op_threads is None:
                # Bulk decode/synthesis: leave one core free so the capture pump and
                # frame hop keep running under a burst (ORT's default pool is every
                # core, spinning while idle). Count the cgroup/taskset budget, not the
                # machine, where the OS can tell them apart.
                affinity = getattr(os, "sched_getaffinity", None)
                cores = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
                intra_op_threads = max(1, cores - 1)
            opts.intra_op_num_threads = intra_op_threads  # frame-path callers pass 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
            kwargs["sess_options"] = opts
            self._sess = onnxruntime.InferenceSession(path, **kwargs)
        else:
            raise ValueError(f"unsupported model type (need .rknn / .onnx): {path}")

    def run(self, inputs: NamedInputs) -> list[Any]:
        """Run inference; outputs in the model's declared order. ORT sessions
        support concurrent ``run``; an RKNN context does NOT, and callers do
        overlap (warmup or the perf-calibration probe vs live decode, eager STT
        vs the utterance worker), so the RKNN branch is serialized per model."""
        if self._rknn is not None:
            with self._rknn_lock:
                if self._released:
                    raise RuntimeError("on-device model released")
                arrs = [arr for _, arr in inputs]
                perm = self._rknn_input_permutation
                if perm is not None:
                    arrs = [
                        self._np.ascontiguousarray(self._np.transpose(arr, perm))
                        if getattr(arr, "ndim", None) == len(perm) else arr
                        for arr in arrs
                    ]
                return self._rknn.inference(inputs=arrs)
        sess = self._sess
        if sess is None:
            raise RuntimeError("on-device model released")
        return sess.run(None, {name: arr for name, arr in inputs})

    def metadata(self) -> dict[str, str]:
        """Embedded key/value metadata (ONNX ``metadata_props``); ``{}`` for RKNN,
        which has none: callers that REQUIRE it should raise clearly so the
        registry can fall back. Modern sherpa-onnx exports carry their whole
        front-end contract here (CMVN stats, LFR params, language ids)."""
        if self._sess is None:
            return {}
        return dict(self._sess.get_modelmeta().custom_metadata_map)

    def input_specs(self) -> list[tuple[str, list, str]]:
        """ONNX input declarations as ``(name, shape, type)``; dims may be symbolic
        strings (e.g. batch ``'N'``). Empty for RKNN, same fallback contract as
        :meth:`metadata`. Lets a stateful streaming model (zipformer's ~35 cache
        tensors) build zero states generically."""
        if self._sess is None:
            return []
        return [(i.name, list(i.shape), i.type) for i in self._sess.get_inputs()]

    def output_names(self) -> list[str]:
        """ONNX output names, in ``run()``'s return order. Empty for RKNN."""
        if self._sess is None:
            return []
        return [o.name for o in self._sess.get_outputs()]

    def input_shape(self, name: str) -> tuple[int, ...] | None:
        """Static shape of input ``name`` for ONNX, else None (RKNN / dynamic /
        absent), the caller then falls back to a known constant."""
        if self._sess is None:
            return None
        for i in self._sess.get_inputs():
            if i.name == name and all(isinstance(d, int) for d in i.shape):
                return tuple(i.shape)
        return None

    def release(self) -> None:
        """Free the NPU context / ORT session; idempotent. Held under
        ``_rknn_lock`` so it can never free the context while a native
        ``inference()`` runs on a to_thread worker (a cancelled to_thread keeps
        running): that would be a use-after-free in C."""
        with self._rknn_lock:
            if self._released:
                return
            self._released = True
            rknn, self._rknn = self._rknn, None
            self._sess = None
            if rknn is not None:
                rknn.release()

    # Construction-time ownership: adapters load models inside a contextlib
    # ExitStack (enter_context each, pop_all() once the adapter owns them), so a
    # failure loading siblings or parsing side files releases every claimed
    # session/NPU core without per-path try/release bookkeeping.
    def __enter__(self) -> OnDeviceModel:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False
