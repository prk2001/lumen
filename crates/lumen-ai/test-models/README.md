# lumen-ai test-models

Pre-generated ONNX fixtures used by `lumen-ai` tests and the
`run_identity_check` self-check.

## `identity.onnx`

A trivial graph: one `Identity` node, input `1×3×4×4` float32 named
`input`, output `1×3×4×4` float32 named `output`. Opset 13, IR 7. Around
135 bytes on disk.

Regenerate with:

```bash
python3 - <<'PY'
import onnx
from onnx import helper, TensorProto

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])
node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
graph = helper.make_graph([node], "identity-graph", [input_tensor], [output_tensor])
opset = helper.make_opsetid("", 13)
model = helper.make_model(graph, producer_name="lumen-ai-test", opset_imports=[opset])
model.ir_version = 7
onnx.checker.check_model(model)
with open("crates/lumen-ai/test-models/identity.onnx", "wb") as f:
    f.write(model.SerializeToString())
PY
```

The crate `include_bytes!`-embeds this file so tests run without
filesystem dependencies.
