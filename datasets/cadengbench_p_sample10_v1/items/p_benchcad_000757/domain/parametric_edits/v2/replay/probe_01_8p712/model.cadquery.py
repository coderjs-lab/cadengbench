import cadquery as cq

result = (
    cq.Workplane("XY")
    .polyline([[0.0, -50.69], [12.6572, -12.6572], [50.69, 0.0], [12.6572, 12.6572], [0.0, 50.69], [-12.6572, 12.6572], [-50.69, 0.0], [-12.6572, -12.6572]])
    .close()
    .extrude(8.712)
)

# Export
show_object(result)
