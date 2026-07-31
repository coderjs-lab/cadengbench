import cadquery as cq

result = (
    cq.Workplane("XY")
    .polygon(6, 18.476)
    .extrude(48.27)
    .faces(">Z").workplane()
    .circle(4.825)
    .extrude(3.86)
    .faces("<Z").workplane()
    .hole(7.389, depth=5.79)
)

# Export
show_object(result)
