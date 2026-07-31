import cadquery as cq

result = (
    cq.Workplane("XY")
    .box(47.817, 6.52, 6.52)
    .edges("|X")
    .chamfer(0.65)
)

# Export
show_object(result)
