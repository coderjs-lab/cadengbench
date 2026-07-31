import cadquery as cq

DEFAULT_PARAMS = {"overall_width": 30.03}
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
    return cq.Workplane('YZ').box(values['overall_width'] * 1.1, 3.22, 3.22)

result = build()
cq.exporters.export(result, "model.step")
