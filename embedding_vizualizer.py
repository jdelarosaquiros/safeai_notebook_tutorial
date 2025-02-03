import marimo

__generated_with = "0.10.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Reactive Runtime""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    x = 2
    x
    return (x,)


@app.cell
def _(x):
    y = 2 + x
    y
    return (y,)


@app.cell
def _(y):
    z = 4 * y
    z 
    return (z,)


@app.cell
def _(mo):
    slider = mo.ui.slider(start=1, stop=10, step=2)
    slider
    return (slider,)


@app.cell
def _(slider):
    e = 2 ** slider.value
    e 
    return (e,)


@app.cell
def _(slider):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Slider

    fig = plt.figure()
    ax = fig.add_subplot(111)

    _x = np.linspace(0, 2*np.pi, 1000)
    amplitude = 1
    _y = amplitude * np.sin(_x)

    line, = ax.plot(_x, _y)
    ax.set_ylim(-5, 5)

    amp = slider.value
    _y = amp * np.sin(_x)
    line.set_ydata(_y)
    ax.set_ylim(-amp*1.1, amp*1.1)
    fig.canvas.draw_idle()

    plt.show()

    return Slider, amp, amplitude, ax, fig, line, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Embedding Visualizer""")
    return


@app.cell
def _():
    import sklearn
    import sklearn.datasets
    import sklearn.manifold

    raw_digits, raw_labels = sklearn.datasets.load_digits(return_X_y=True)
    return raw_digits, raw_labels, sklearn


@app.cell
def _(pd, raw_digits, raw_labels, sklearn):
    X_embedded = sklearn.decomposition.PCA(
        n_components=2, whiten=True
    ).fit_transform(raw_digits)

    embedding = pd.DataFrame(
        {"x": X_embedded[:, 0], "y": X_embedded[:, 1], "digit": raw_labels}
    ).reset_index()
    return X_embedded, embedding


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
        Here's a PCA **embedding of numerical digits**: each point represents a 
        digit, with similar digits close to each other. The data is from the UCI 
        ML handwritten digits dataset.

        This notebook will automatically drill down into points you **select with 
        your mouse**; try it!
        """
    )
    return


@app.cell
def _(embedding, mo, scatter):
    chart = mo.ui.altair_chart(scatter(embedding))
    chart
    return (chart,)


@app.cell
def _(chart, mo):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell(hide_code=True)
def _(chart, mo, raw_digits, table):
    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table
    mo.stop(not len(chart.value))

    def show_images(indices, max_images=10):
        import matplotlib.pyplot as plt

        indices = indices[:max_images]
        images = raw_digits.reshape((-1, 8, 8))[indices]
        fig, axes = plt.subplots(1, len(indices))
        fig.set_size_inches(12.5, 1.5)
        if len(indices) > 1:
            for im, ax in zip(images, axes.flat):
                ax.imshow(im, cmap="gray")
                ax.set_yticks([])
                ax.set_xticks([])
        else:
            axes.imshow(images[0], cmap="gray")
            axes.set_yticks([])
            axes.set_xticks([])
        plt.tight_layout()
        return fig

    selected_images = (
        show_images(list(chart.value["index"]))
        if not len(table.value)
        else show_images(list(table.value["index"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table}
        """
    )
    return selected_images, show_images


@app.cell
def _(embedding, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM embedding
        """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(alt):
    def scatter(df):
        return (alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x:Q").scale(domain=(-2.5, 2.5)),
            y=alt.Y("y:Q").scale(domain=(-2.5, 2.5)),
            color=alt.Color("digit:N"),
        ).properties(width=500, height=500))
    return (scatter,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip
        await micropip.install("altair")

    import altair as alt
    return alt, micropip, sys


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
