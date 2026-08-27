"""A minimal RKNN / ONNX model runtime shared by the on-device adapters, mirroring the
``init_model`` / ``run_*`` / ``release_model`` pattern of the Rockchip rknn_model_zoo
demos they are ported from. Backend follows the file extension: ``.rknn`` (Rockchip NPU)
and ``.onnx`` (``onnxruntime``) are the only supported artifacts.

``.onnx`` execution providers are configurable, so the SAME export runs on CPU (default)
or an accelerator (Jetson: ``["TensorrtExecutionProvider", "CUDAExecutionProvider",
"CPUExecutionProvider"]``); always keep ``CPUExecutionProvider`` last as the fallback for
unsupported nodes. ``provider_options`` is the parallel per-provider option list.

Heavy runtimes import lazily; a construction failure is raised to the registry, which
warns and falls back to a cloud/system backend.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from loguru import logger

# (name, array) pairs: ONNX is fed by name, RKNN positionally, so this carries both.
NamedInputs = list[tuple[str, Any]]


def _core_mask(api: Any, core_mask: str) -> Any:
    """Resolve ``core_mask`` to the runtime's ``NPU_CORE_*`` constant (RK3588 has 3
    cores; AUTO lets the runtime decide). Older toolkits lack some names: warn rather
    than silently run on one core."""
    name = f"NPU_CORE_{core_mask.upper()}"
    mask = getattr(api, name, None)
    if mask is None:  # NPU_CORE_AUTO == 0, so this must test None, not falsiness
        logger.warning("voice: this RKNN runtime has no {}; using NPU_CORE_AUTO", name)
        return api.NPU_CORE_AUTO
    return mask


def _load_rknn(path: str, *, core_mask: str, target: str | None, device_id: str | None) -> Any:
    """Load a ``.rknn`` model, preferring ``rknn-toolkit-lite2`` (the on-board runtime,
    ``[rknn]`` extra, aarch64-only) and falling back to the full ``rknn-toolkit2``; the
    toolkits differ only in ``init_runtime``, where the full one also takes
    ``target=``/``device_id=``."""
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
        # Drop lite2's copy of the model file: dead weight once init_runtime loaded the
        # NPU (487 MB for SenseVoice). getattr: absent on the full toolkit.
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


def check_deterministic(model: Any, inputs: NamedInputs) -> None:
    """Run *inputs* through *model* twice and demand the same answer (construction
    probes only).

    These graphs are deterministic by contract, so a real difference means the runtime
    reads uninitialized memory — measured on onnxruntime 1.27.0 for the zipformer encoder
    and the openWakeWord embedding, which decode to silence rather than raising. The
    tolerance clears GPU reduction jitter (~1e-6 on the CUDA EP) while catching that
    corruption, 3-4 orders of magnitude larger.
    """
    import numpy as np

    for first, second in zip(model.run(inputs), model.run(inputs), strict=True):
        if not np.allclose(np.asarray(first), np.asarray(second), rtol=1e-3, atol=1e-5):
            raise RuntimeError(
                "this onnxruntime/RKNN build returns different output for identical "
                "input -- on-device inference would be silent garbage. onnxruntime "
                "1.27.0 is known-bad; install !=1.27.0"
            )


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
        profile: str = "frame",
        prepack: bool | None = None,
    ):
        """``profile`` (ONNX only): ``"frame"`` = ORT defaults, for fixed-shape
        per-frame sessions. ``"bulk"`` = per-utterance sessions: the CPU arena is off
        (it ratchets to the largest input ever seen and never shrinks) and so is
        weight pre-packing, unless ``prepack=True`` keeps it (int8 GEMMs decode
        markedly slower unpacked; fp32 loses nothing)."""
        self._rknn: Any = None
        self._sess: Any = None
        self._released = False
        self._rknn_lock = threading.Lock()  # RKNN contexts are not thread-safe
        # Pre-transpose RKNN inputs of matching rank into the import's reported layout;
        # declaring data_format per call instead makes Lite repair-and-WARN every
        # inference.
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
                # frame hop survive a burst (ORT defaults to every core, spinning idle).
                # Count the cgroup/taskset budget, not the machine.
                affinity = getattr(os, "sched_getaffinity", None)
                cores = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
                intra_op_threads = max(1, cores - 1)
            opts.intra_op_num_threads = intra_op_threads  # frame-path callers pass 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
            if profile == "bulk":
                opts.enable_cpu_mem_arena = False
                if prepack is None or not prepack:
                    opts.add_session_config_entry("session.disable_prepacking", "1")
            kwargs["sess_options"] = opts
            self._sess = onnxruntime.InferenceSession(path, **kwargs)
        else:
            raise ValueError(f"unsupported model type (need .rknn / .onnx): {path}")

    def run(self, inputs: NamedInputs) -> list[Any]:
        """Run inference; outputs in the model's declared order. ORT sessions support
        concurrent ``run``; an RKNN context does NOT and callers do overlap (warmup vs
        live decode, eager STT vs utterance worker), so RKNN is serialized per model."""
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
        """Embedded key/value metadata (ONNX ``metadata_props``); ``{}`` for RKNN, which
        has none — callers that REQUIRE it should raise so the registry can fall back.
        sherpa-onnx exports carry their whole front-end contract here (CMVN, LFR, langs)."""
        if self._sess is None:
            return {}
        return dict(self._sess.get_modelmeta().custom_metadata_map)

    def input_specs(self) -> list[tuple[str, list, str]]:
        """ONNX input declarations as ``(name, shape, type)``; dims may be symbolic strings
        (batch ``'N'``). Empty for RKNN, same fallback contract as :meth:`metadata`. Lets
        a stateful streaming model (zipformer's ~35 caches) build zero states
        generically."""
        if self._sess is None:
            return []
        return [(i.name, list(i.shape), i.type) for i in self._sess.get_inputs()]

    def output_names(self) -> list[str]:
        """ONNX output names, in ``run()``'s return order. Empty for RKNN."""
        if self._sess is None:
            return []
        return [o.name for o in self._sess.get_outputs()]

    def input_shape(self, name: str) -> tuple[int, ...] | None:
        """Static shape of input ``name`` for ONNX, else None (RKNN / dynamic / absent);
        the caller then falls back to a known constant."""
        if self._sess is None:
            return None
        for i in self._sess.get_inputs():
            if i.name == name and all(isinstance(d, int) for d in i.shape):
                return tuple(i.shape)
        return None

    def release(self) -> None:
        """Free the NPU context / ORT session; idempotent. Held under ``_rknn_lock``: freeing
        while a native ``inference()`` runs on a to_thread worker (which keeps running
        after cancellation) is a use-after-free in C."""
        with self._rknn_lock:
            if self._released:
                return
            self._released = True
            rknn, self._rknn = self._rknn, None
            self._sess = None
            if rknn is not None:
                rknn.release()

    # Construction-time ownership: adapters load models inside an ExitStack
    # (enter_context each, pop_all() once the adapter owns them), so a failure loading
    # siblings releases every claimed session/NPU core.
    def __enter__(self) -> OnDeviceModel:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False
