import cadquery as cq

result = (
    cq.Workplane("YZ")
    .box(27.027, 3.22, 3.22)
)

# Export
show_object(result)
