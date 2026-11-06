"""Shared look and feel: the pastel blue theme and small page-building helpers."""

from __future__ import annotations

import html

import streamlit as st

from utils.preprocessing import PROJECT_ROOT

THEME_CSS = """
<style>
:root {
  --ice: #F3F8FC; --pastel: #B8D8F0; --powder: #D6EAF8; --baby: #C5E3F6;
  --periwinkle: #D9DDF7; --navy: #344B64; --grey: #71869A; --cyan: #DDF3F5;
  --shadow: 0 6px 22px rgba(95, 147, 196, 0.12);
  --shadow-hover: 0 10px 28px rgba(95, 147, 196, 0.20);
}

/* Page background: one quiet gradient, nothing competing with the content. */
.stApp { background: linear-gradient(180deg, #EAF4FB 0%, #F3F8FC 260px, #FFFFFF 100%); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1240px; }

h1, h2, h3, h4 { color: var(--navy) !important; letter-spacing: -0.015em; }
h1 { font-weight: 650 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #EAF4FB 0%, #F3F8FC 100%);
  border-right: 1px solid var(--powder);
}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {
  border-radius: 10px; transition: background-color .18s ease;
}
[data-testid="stSidebarNavLink"]:hover { background: var(--powder); }
[data-testid="stSidebarHeader"] img { height: 2.6rem; max-width: 100%; }
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: #FFFFFF; box-shadow: var(--shadow);
}

/* Cards: containers made with card(), and metrics */
[class*="st-key-card_"], [data-testid="stForm"] {
  background: #FFFFFF; border: 1px solid #E4F0F9 !important; border-radius: 18px !important;
  box-shadow: var(--shadow); padding: 20px 22px !important;
}
[data-testid="stMetric"] {
  background: rgba(255, 255, 255, 0.78); backdrop-filter: blur(8px);
  border: 1px solid #E4F0F9 !important; border-radius: 16px; padding: 16px 18px;
  box-shadow: var(--shadow); transition: box-shadow .18s ease, transform .18s ease;
}
[data-testid="stMetric"]:hover { box-shadow: var(--shadow-hover); transform: translateY(-1px); }
[data-testid="stMetricLabel"] p { color: var(--grey) !important; font-size: .82rem; font-weight: 500; }
[data-testid="stMetricValue"] { color: var(--navy) !important; font-weight: 650; font-size: 1.75rem; }

/* Buttons: pastel fill, navy text, both on primary and secondary */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  background: #EAF4FB; color: var(--navy); border: 1px solid var(--pastel);
  font-weight: 550; transition: background-color .18s ease, box-shadow .18s ease;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
  background: var(--pastel); color: var(--navy); border: 1px solid #A5CCE9;
}
.stButton > button p, .stDownloadButton > button p, .stFormSubmitButton > button p { color: var(--navy); }
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  background: var(--baby); color: var(--navy); border-color: #A5CCE9;
  box-shadow: 0 4px 14px rgba(95, 147, 196, 0.22);
}

/* Tabs, charts and tables */
[data-baseweb="tab-list"] { gap: 6px; }
[data-baseweb="tab"] { border-radius: 10px 10px 0 0; }
[data-testid="stPlotlyChart"] { border-radius: 14px; overflow: hidden; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 14px; }

/* Custom blocks */
.sr-hero {
  padding: 34px 38px; border-radius: 24px; margin-bottom: 8px;
  background: linear-gradient(120deg, rgba(214,234,248,.9), rgba(217,221,247,.65) 60%, rgba(221,243,245,.8));
  border: 1px solid rgba(255,255,255,.8); box-shadow: var(--shadow);
}
.sr-eyebrow { font-size: .76rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: #5F7E9C; }
.sr-hero h1 { margin: 6px 0 8px; padding: 0; font-size: 2.35rem; line-height: 1.16; }
.sr-hero p { margin: 0; max-width: 780px; color: #4A6078; font-size: 1.04rem; line-height: 1.55; }
.sr-page { margin-bottom: 10px; }
.sr-page h1 { margin: 2px 0 4px; padding: 0; font-size: 1.95rem; }
.sr-page p { margin: 0; color: var(--grey); max-width: 820px; line-height: 1.5; }
.sr-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.sr-step {
  background: #FFFFFF; border: 1px solid #E4F0F9; border-radius: 16px; padding: 16px;
  box-shadow: var(--shadow);
}
.sr-step b { display: block; color: var(--navy); margin-bottom: 4px; font-weight: 600; }
.sr-step span { color: var(--grey); font-size: .88rem; line-height: 1.42; display: block; }
.sr-step i { font-style: normal; color: #5F93C4; font-size: .75rem; font-weight: 600; letter-spacing: .06em; }
.sr-chip {
  display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem;
  background: var(--cyan); color: var(--navy); margin: 2px 6px 2px 0;
}
.sr-chip.alt { background: var(--periwinkle); }
</style>
"""


def apply_theme() -> None:
    """Inject the stylesheet. Called once, from app.py, on every rerun."""
    st.session_state["_card_count"] = 0
    st.html(THEME_CSS)


def sidebar_brand() -> None:
    st.logo(str(PROJECT_ROOT / "assets" / "logo.svg"), size="large")


def card(stretch: bool = False):
    """A white rounded card. The key is what the stylesheet attaches to."""
    count = st.session_state.get("_card_count", 0) + 1
    st.session_state["_card_count"] = count
    return st.container(key=f"card_{count}", height="stretch" if stretch else "content")


def hero(eyebrow: str, title: str, text: str) -> None:
    st.html(
        f'<div class="sr-hero"><div class="sr-eyebrow">{html.escape(eyebrow)}</div>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></div>"
    )


def page_header(title: str, text: str) -> None:
    st.html(f'<div class="sr-page"><h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></div>')


def flow(steps: list[tuple[str, str]]) -> None:
    """A numbered row of cards. Wraps on narrow screens."""
    cards = "".join(
        f'<div class="sr-step"><i>STEP {n}</i><b>{html.escape(head)}</b>'
        f"<span>{html.escape(body)}</span></div>"
        for n, (head, body) in enumerate(steps, start=1)
    )
    st.html(f'<div class="sr-flow">{cards}</div>')


def chips(labels: list[str]) -> None:
    items = "".join(
        f'<span class="sr-chip{" alt" if i % 2 else ""}">{html.escape(label)}</span>'
        for i, label in enumerate(labels)
    )
    st.html(f"<div>{items}</div>")


def chart(fig, key: str) -> None:
    """Draw a Plotly figure with our template, not Streamlit's."""
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False})
