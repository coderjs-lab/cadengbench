import cadquery as cq

result = (
    cq.Workplane("XZ")
    .box(21.195, 2.36, 2.36)
    .edges("|X")
    .chamfer(0.35)
    .faces(">Y").workplane()
    .slot2D(18.84, 0.71, 0)
    .cutBlind(-0.47)
)

# Export
show_object(result)
