import cadquery as cq

DEFAULT_PARAMS = {"overall_height": 20.32}
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
            cq.Edge.makeLine((0.0, 17.78, 0.0), (0.0, 0.0, 0.0)),
            cq.Edge.makeLine((0.0, 0.0, 0.0), (170.18, 0.0, 0.0)),
            cq.Edge.makeLine((170.18, 0.0, 0.0), (170.18, 5.08, 0.0)),
            cq.Edge.makeLine((170.18, 5.08, 0.0), (187.96, 5.08, 0.0)),
            cq.Edge.makeLine((187.96, 5.08, 0.0), (187.96, 17.78, 0.0)),
            cq.Edge.makeLine((187.96, 17.78, 0.0), (78.74, 17.78, 0.0)),
            cq.Edge.makeLine((78.74, 17.78, 0.0), (78.74, 12.7, 0.0)),
            cq.Edge.makeLine((78.74, 12.7, 0.0), (20.32, 12.7, 0.0)),
            cq.Edge.makeLine((20.32, 12.7, 0.0), (20.32, 17.78, 0.0)),
            cq.Edge.makeLine((20.32, 17.78, 0.0), (0.0, 17.78, 0.0))
    ])
    solid_0_0 = cq.Solid.extrudeLinear(wire_0_0_0, [], cq.Vector(*tuple(component * (values['overall_height'] * 1.1) for component in (0.0, 0.0, 1.0))))
    tool_0 = solid_0_0
    if current is None:
        current = tool_0
    else:
        current = cq.Compound.makeCompound([current, tool_0])
    return current

result = build()
cq.exporters.export(result, "model.step")
