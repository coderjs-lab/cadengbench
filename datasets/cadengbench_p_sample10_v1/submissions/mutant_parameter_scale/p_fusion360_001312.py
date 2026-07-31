import cadquery as cq

DEFAULT_PARAMS = {"overall_width": 300.0}
PARAMETER_MANIFEST = {
    "P-001": {
        "name": 'overall_width',
        "unit": "mm",
        "role": "principal_dimension",
    }
}

def build(params=None):
    values = dict(DEFAULT_PARAMS)
    if params is not None:
        values.update(params)
    current = None
    wire_0_0_0 = cq.Wire.assembleEdges([
            cq.Edge.makeCircle(35.0, (525.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ])
    wire_0_0_1 = cq.Wire.assembleEdges([
            cq.Edge.makeCircle(15.0, (525.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ])
    solid_0_0 = cq.Solid.extrudeLinear(wire_0_0_0, [wire_0_0_1], cq.Vector(*tuple(component * (values['overall_width'] * 1.1) for component in (0.0, 1.0, 0.0))))
    wire_0_1_0 = cq.Wire.assembleEdges([
            cq.Edge.makeCircle(15.0, (525.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ])
    solid_0_1 = cq.Solid.extrudeLinear(wire_0_1_0, [], cq.Vector(*tuple(component * (values['overall_width'] * 1.1) for component in (0.0, 1.0, 0.0))))
    tool_0 = solid_0_0
    tool_0 = tool_0.fuse(solid_0_1)
    if current is None:
        current = tool_0
    else:
        current = cq.Compound.makeCompound([current, tool_0])
    wire_1_0_0 = cq.Wire.assembleEdges([
            cq.Edge.makeCircle(35.0, (525.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ])
    wire_1_0_1 = cq.Wire.assembleEdges([
            cq.Edge.makeCircle(15.0, (525.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ])
    solid_1_0 = cq.Solid.extrudeLinear(wire_1_0_0, [wire_1_0_1], cq.Vector(*tuple(component * (values['overall_width'] * 1.1 + 30.0) for component in (0.0, 1.0, 0.0))))
    tool_1 = solid_1_0
    current = current.cut(tool_1)
    return current

result = build()
cq.exporters.export(result, "model.step")
