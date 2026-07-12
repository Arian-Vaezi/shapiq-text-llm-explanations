"""Plot loading, polishing, and fallback rendering."""

# ruff: noqa: E402, F405

from __future__ import annotations

# Mechanical re-export chain preserves the monolith's shared global namespace.
from .shapley import *  # noqa: F403

# Preserve entrypoint-relative path semantics in mechanically moved functions.
__file__ = str(Path(__file__).parent.parent / "app.py")


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
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="x", color="#d7dfdf", alpha=0.65, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for patch in ax.patches:
        width = patch.get_width()
        if abs(width) < 0.01:
            continue
        x_pos = width + (0.015 if width >= 0 else -0.015)
        ha = "left" if width >= 0 else "right"
        # Direction is marked with both an arrow glyph and a color (never color
        # alone), so sign reads correctly in black-and-white print or for
        # color-vision-deficient readers -- the bar fill itself keeps the
        # shapiq library's own red/blue convention untouched.
        arrow, label_color = ("↑", "#197a52") if width >= 0 else ("↓", "#b3261e")
        ax.text(
            x_pos,
            patch.get_y() + patch.get_height() / 2,
            f"{arrow} {width:.2f}",
            va="center",
            ha=ha,
            fontsize=7,
            color=label_color,
        )

    fig.tight_layout()
    return fig


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
