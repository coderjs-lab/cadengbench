import cadquery as cq

DEFAULT_PARAMS = {"overall_height": 7.92}
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
    return cq.Workplane('XY').polyline([[0.0, -50.69], [12.6572, -12.6572], [50.69, 0.0], [12.6572, 12.6572], [0.0, 50.69], [-12.6572, 12.6572], [-50.69, 0.0], [-12.6572, -12.6572]]).close().extrude(values['overall_height'])

result = build()
cq.exporters.export(result, "model.step")
