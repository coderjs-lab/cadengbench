import cadquery as cq

DEFAULT_PARAMS = {"overall_length": 23.55}
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
    return cq.Workplane('XZ').box(values['overall_length'], 2.36, 2.36).edges('|X').chamfer(0.35).faces('>Y').workplane().slot2D(18.84, 0.71, 0).cutBlind(-0.47)

result = build()
cq.exporters.export(result, "model.step")
