from matplotlib.testing.decorators import image_comparison

from nanomesh.mpl.meshplot import plot_line_triangle


@image_comparison(
    baseline_images=['line_triangle_plot'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_triangle_mesh_plot(mesh_square2d):
    plot_line_triangle(mesh_square2d, label=None)
