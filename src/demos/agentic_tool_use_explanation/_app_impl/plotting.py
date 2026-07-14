"""Plot loading, polishing, and fallback rendering."""

# ruff: noqa: E402, F405

from __future__ import annotations

import numpy as np

from shapiq.plot._config import BLUE, RED

# Mechanical re-export chain preserves the monolith's shared global namespace.
from .shapley import *  # noqa: F403

# Preserve entrypoint-relative path semantics in mechanically moved functions.
__file__ = str(Path(__file__).parent.parent / "app.py")

HEATMAP_LARGE_ANNOTATION_MAX_PLAYERS = 6
HEATMAP_ANNOTATION_MAX_PLAYERS = 10
HEATMAP_LARGE_ANNOTATION_FONTSIZE = 8.0
HEATMAP_MEDIUM_ANNOTATION_FONTSIZE = 5.5
HEATMAP_DARK_CELL_THRESHOLD = 0.55
BAR_ANNOTATION_FONTSIZE = 6.5
BAR_ANNOTATION_OFFSET_FRACTION = 0.02
BAR_ANNOTATION_GUTTER_FACTOR = 1.18


def polish_bar(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    xlabel: str = "Target-tool attribution",
    title: str = "",
    figsize: tuple[float, float] = (5.2, 2.6),
) -> plt.Figure:
    """Make package bar plot fit a compact Streamlit layout (e.g. one of two side-by-side columns).

    ``title`` defaults to empty: the Streamlit section header directly above the plot
    (e.g. "2. Individual segment effects -- SV") already states what the plot is, so an
    internal matplotlib title would just repeat it. Pass a non-empty ``title`` to opt
    back into an in-plot title for a call site without its own section header.
    """
    fig.set_size_inches(*figsize)
    ax.set_title("", loc="center")
    if title:
        ax.set_title(title, loc="left", fontsize=9, pad=5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="x", color="#d7dfdf", alpha=0.65, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    visible_widths = [patch.get_width() for patch in ax.patches if abs(patch.get_width()) >= 0.01]
    if visible_widths:
        left_limit, right_limit = ax.get_xlim()
        axis_extent = max(
            abs(left_limit),
            abs(right_limit),
            *(abs(width) for width in visible_widths),
        )
        # Provisional gutter: a cheap percentage-of-value estimate, sufficient for
        # typical Shapley magnitudes. Not guaranteed to fit the annotation text for
        # every value -- see _expand_xlim_to_fit_annotations below, which measures
        # the real rendered text and widens this further only if actually needed.
        provisional_extent = axis_extent * BAR_ANNOTATION_GUTTER_FACTOR
        ax.set_xlim(-provisional_extent, provisional_extent)
        annotation_offset = axis_extent * BAR_ANNOTATION_OFFSET_FRACTION
    else:
        annotation_offset = 0.0

    annotations: list[tuple[plt.Text, str]] = []
    for patch in ax.patches:
        width = patch.get_width()
        if abs(width) < 0.01:
            continue
        x_pos = width + (annotation_offset if width >= 0 else -annotation_offset)
        ha = "left" if width >= 0 else "right"
        # Direction is marked with both an arrow glyph and a color (never color
        # alone), so sign reads correctly in black-and-white print or for
        # color-vision-deficient readers. The label color reuses the same
        # shapiq red/blue convention as the bar fill itself (red = positive/
        # support, blue = negative/oppose), so the annotation never disagrees
        # with the bar it is labeling.
        arrow = "↑" if width >= 0 else "↓"
        label_color = RED.hex if width >= 0 else BLUE.hex
        text = ax.text(
            x_pos,
            patch.get_y() + patch.get_height() / 2,
            f"{arrow} {width:+.2f}",
            va="center",
            ha=ha,
            fontsize=BAR_ANNOTATION_FONTSIZE,
            color=label_color,
            clip_on=True,
        )
        annotations.append((text, ha))

    if annotations:
        _expand_xlim_to_fit_annotations(fig, ax, annotations)

    fig.tight_layout(pad=0.8)
    fig.subplots_adjust(left=max(fig.subplotpars.left, 0.24))
    return fig


def _expand_xlim_to_fit_annotations(
    fig: plt.Figure,
    ax: plt.Axes,
    annotations: list[tuple[plt.Text, str]],
    *,
    safety_factor: float = 1.02,
) -> None:
    """Widen xlim just enough that every annotation's actual rendered extent fits.

    The percentage-based gutter set by the caller is a cheap estimate, not a
    guarantee: it can undershoot once an annotation's fixed point-size text no
    longer fits inside a percentage of a large bar's own value (e.g. a Shapley
    value >= ~1.0 gets visibly truncated under ``clip_on=True`` while smaller
    values render fine). This measures each annotation's real window extent
    after a draw and expands xlim only as far as actually needed -- never more,
    never less -- so text is never silently clipped regardless of magnitude,
    and plots that already fit are left with their existing xlim unchanged.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse_transform = ax.transData.inverted()
    _, current_right = ax.get_xlim()
    max_extent_needed = current_right
    for text, ha in annotations:
        window_extent = text.get_window_extent(renderer)
        far_x_pixels = window_extent.x1 if ha == "left" else window_extent.x0
        ((far_x_data, _unused_y),) = inverse_transform.transform([(far_x_pixels, window_extent.y0)])
        max_extent_needed = max(max_extent_needed, abs(far_x_data))

    final_extent = max_extent_needed * safety_factor
    ax.set_xlim(-final_extent, final_extent)
    fig.canvas.draw()


def polish_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    segments: list[ToolUseSegment],
    *,
    title: str = "",
    figsize: tuple[float, float] = (4.3, 3.2),
) -> plt.Figure:
    """Make package heatmap fit a compact Streamlit layout (e.g. one of two side-by-side columns).

    ``title`` defaults to empty: the Streamlit section header directly above the plot
    (e.g. "3. Pairwise interactions -- k-SII") already states what the plot is, so an
    internal matplotlib title would just repeat it. Pass a non-empty ``title`` to opt
    back into an in-plot title for a call site without its own section header.
    """
    fig.set_size_inches(*figsize)
    ax.set_title("", loc="center")
    if title:
        ax.set_title(title, loc="left", fontsize=9, pad=5)
    ax.tick_params(axis="x", labelrotation=30, labelsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)

    n_players = len(segments)
    if ax.images:
        image = ax.images[0]
        matrix = np.asarray(np.ma.getdata(image.get_array()), dtype=float)
        if matrix.shape == (n_players, n_players):
            max_abs_value = float(np.max(np.abs(matrix))) if matrix.size else 0.0
            color_limit = max_abs_value if max_abs_value > 0 else 1.0
            image.set_clim(-color_limit, color_limit)
            image.set_data(matrix)

            annotation_fontsize = None
            if n_players <= HEATMAP_LARGE_ANNOTATION_MAX_PLAYERS:
                annotation_fontsize = HEATMAP_LARGE_ANNOTATION_FONTSIZE
            elif n_players <= HEATMAP_ANNOTATION_MAX_PLAYERS:
                annotation_fontsize = HEATMAP_MEDIUM_ANNOTATION_FONTSIZE

            for annotation in list(ax.texts):
                x_position, y_position = annotation.get_position()
                column = int(round(x_position))
                row = int(round(y_position))
                is_matrix_annotation = (
                    0 <= row < n_players
                    and 0 <= column < n_players
                    and abs(x_position - column) < 1e-9
                    and abs(y_position - row) < 1e-9
                )
                if not is_matrix_annotation:
                    continue
                if annotation_fontsize is None:
                    annotation.remove()
                    continue
                intensity = abs(float(matrix[row, column])) / color_limit
                annotation.set_color(
                    "white" if intensity >= HEATMAP_DARK_CELL_THRESHOLD else "#1f1f1f"
                )
                annotation.set_fontsize(annotation_fontsize)
                annotation.set_clip_on(True)

    if segments:
        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                len(segments),
                len(segments),
                fill=False,
                edgecolor="#b15d3b",
                linewidth=2.2,
                zorder=5,
            )
        )
    fig.tight_layout()
    return fig


def load_text_plotters() -> tuple[object | None, object | None, str | None]:
    """Load text plotting helpers without importing the full shapiq plotting package."""
    try:
        module = load_sentence_plot_module()
    except Exception as error:  # noqa: BLE001
        return None, None, str(error)
    return (
        module.token_attribution_bar_plot,
        module.sentence_interaction_heatmap,
        None,
    )


def load_sentence_plot_module() -> types.ModuleType:
    """Load shapiq.plot.sentence directly to avoid optional tree C extensions."""
    plot_dir = Path(__file__).resolve().parents[2] / "shapiq" / "plot"
    package = sys.modules.get(TEXT_PLOT_PACKAGE)
    if package is None:
        package = types.ModuleType(TEXT_PLOT_PACKAGE)
        package.__path__ = [str(plot_dir)]  # type: ignore[attr-defined]
        sys.modules[TEXT_PLOT_PACKAGE] = package

    config_name = f"{TEXT_PLOT_PACKAGE}._config"
    if config_name not in sys.modules:
        config_spec = importlib.util.spec_from_file_location(
            config_name,
            plot_dir / "_config.py",
        )
        if config_spec is None or config_spec.loader is None:
            msg = "Could not load shapiq sentence plot color configuration."
            raise ImportError(msg)
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules[config_name] = config_module
        config_spec.loader.exec_module(config_module)

    sentence_name = f"{TEXT_PLOT_PACKAGE}.sentence"
    sentence_module = sys.modules.get(sentence_name)
    if sentence_module is not None:
        return sentence_module

    sentence_spec = importlib.util.spec_from_file_location(
        sentence_name,
        plot_dir / "sentence.py",
    )
    if sentence_spec is None or sentence_spec.loader is None:
        msg = "Could not load shapiq sentence plotting helpers."
        raise ImportError(msg)
    sentence_module = importlib.util.module_from_spec(sentence_spec)
    sys.modules[sentence_name] = sentence_module
    sentence_spec.loader.exec_module(sentence_module)
    return sentence_module


def show_fallback_attribution_chart(attribution_frame: pd.DataFrame) -> None:
    """Render a small Streamlit-native attribution chart when matplotlib plots are unavailable."""
    if attribution_frame.empty:
        st.info("No first-order attributions are available to plot.")
        return
    chart_frame = attribution_frame[["segment", "attribution"]].copy()
    chart_frame = chart_frame.sort_values("attribution").set_index("segment")
    st.bar_chart(chart_frame, use_container_width=True)


def show_fallback_interaction_table(pairwise_matrix: pd.DataFrame, labels: list[str]) -> None:
    """Render pairwise interactions as a table when the heatmap helper is unavailable."""
    fallback_matrix = pairwise_matrix.copy()
    fallback_matrix.index = labels
    fallback_matrix.columns = labels
    st.dataframe(fallback_matrix, use_container_width=True)


def display_demo_path() -> str:
    """Return a stable demo path caption for Streamlit and test runners."""
    demo_path = Path(__file__).resolve().parent
    try:
        return str(demo_path.relative_to(Path.cwd()))
    except ValueError:
        return str(demo_path)


# Resolve the original monolith's forward reference without changing either body.
from . import shapley as _shapley

_shapley.load_sentence_plot_module = load_sentence_plot_module

__all__ = [name for name in globals() if not name.startswith("__")]
