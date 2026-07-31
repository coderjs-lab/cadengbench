import cadquery as cq

DEFAULT_PARAMS = {"overall_length": 43.47}
PARAMETER_MANIFEST = {
    "P-001": {
        "name": 'overall_length',
        "unit": "mm",
        "role": "principal_dimension",
    }
}

def build(params=None):
    values = dict(DEFAULT_PARAMS)
    if params is not None:
        values.update(params)
    return cq.Workplane('XY').box(values['overall_length'] * 1.1, 6.52, 6.52).edges('|X').chamfer(0.65)

result = build()
cq.exporters.export(result, "model.step")
