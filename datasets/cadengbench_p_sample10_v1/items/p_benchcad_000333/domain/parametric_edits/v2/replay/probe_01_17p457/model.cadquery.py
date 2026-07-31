import cadquery as cq

result = (
    cq.Workplane("XZ")
    .polygon(6, 27.714)
    .extrude(39.66)
    .faces(">Y").workplane()
    .circle(9.065)
    .extrude(4.53)
    .edges(">Y")
    .chamfer(0.4)
    .faces("<Y").workplane()
    .hole(17.457, depth=6.8)
)

# Export
show_object(result)
