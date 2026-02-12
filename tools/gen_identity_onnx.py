from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "assets" / "models" / "identity.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # y = x, float32 [1,3]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])

    node = helper.make_node("Identity", inputs=["x"], outputs=["y"], name="Identity")

    graph = helper.make_graph(
        [node],
        "identity_graph",
        [x],
        [y],
        initializer=[],
    )

    model = helper.make_model(
        graph,
        producer_name="staticplay-sd1-mobile",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    # Keep IR low for compatibility with older ONNX Runtime builds.
    model.ir_version = 7

    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    print(f"Wrote: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
