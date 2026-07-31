import cadquery as cq

DEFAULT_PARAMS = {"opening_group_001_diameter": 15.87}
PARAMETER_MANIFEST = {
    "P-001": {
        "name": 'opening_group_001_diameter',
        "unit": "mm",
        "role": "principal_dimension",
    }
}

def build(params=None):
    values = dict(DEFAULT_PARAMS)
    if params is not None:
        values.update(params)
    return cq.Workplane('XZ').polygon(6, 27.714).extrude(39.66).faces('>Y').workplane().circle(9.065).extrude(4.53).edges('>Y').chamfer(0.4).faces('<Y').workplane().hole(values['opening_group_001_diameter'] * 1.1, depth=6.8)

result = build()
cq.exporters.export(result, "model.step")
