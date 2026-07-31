import cadquery as cq

result = (
    cq.Workplane("YZ")
    .box(30.03, 3.22, 3.22)
)

# Export
show_object(result)
