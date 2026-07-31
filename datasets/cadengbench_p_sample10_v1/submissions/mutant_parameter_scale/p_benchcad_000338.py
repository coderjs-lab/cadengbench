import cadquery as cq

DEFAULT_PARAMS = {"opening_group_001_diameter": 8.21}
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
    return cq.Workplane('XY').polygon(6, 18.476).extrude(48.27).faces('>Z').workplane().circle(4.825).extrude(3.86).faces('<Z').workplane().hole(values['opening_group_001_diameter'] * 1.1, depth=5.79)

result = build()
cq.exporters.export(result, "model.step")
