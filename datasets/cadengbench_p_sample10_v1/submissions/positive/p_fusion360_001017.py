import cadquery as cq

DEFAULT_PARAMS = {"overall_height": 55.0}
PARAMETER_MANIFEST = {
    "P-001": {
        "name": 'overall_height',
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
            cq.Edge.makeCircle(7.5, (-100.0, 30.0, 45.0), (0.0, 0.0, 1.0))
    ])
    solid_0_0 = cq.Solid.extrudeLinear(wire_0_0_0, [], cq.Vector(*tuple(component * (values['overall_height'] * 1.0) for component in (0.0, 0.0, 1.0))))
    tool_0 = solid_0_0
    if current is None:
        current = tool_0
    else:
        current = cq.Compound.makeCompound([current, tool_0])
    return current

result = build()
cq.exporters.export(result, "model.step")
