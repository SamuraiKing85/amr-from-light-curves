"""
gui/theme.py  –  Startup-only theme + user preferences
========================================================
Theme is chosen ONCE at startup from saved config (default: light).
No runtime widget-walking — keeps every tab simple and fast.

Preferences (theme, last file paths, window geometry) are persisted
to a small JSON file next to the application.
"""

import json, os, tkinter as tk
from tkinter import ttk
from pathlib import Path

# --- Font constants -----------------------------------------------------------
FONT         = "Segoe UI"
FONT_MONO    = "Consolas"
FONT_HEADING = (FONT, 11, "bold")
FONT_BODY    = (FONT, 9)
FONT_SMALL   = (FONT, 8)
FONT_SMALL_B = (FONT, 8, "bold")
FONT_TINY    = (FONT, 7)
FONT_MONO_SM = (FONT_MONO, 9)

# --- Palettes -----------------------------------------------------------------
_LIGHT = dict(
    bg="#f5f6fa", bg_card="#ffffff", bg_input="#ffffff",
    bg_terminal="#1e1e1e", bg_plot="#ffffff", bg_plot_ax="#fafafa",
    fg="#2c3e50", fg_dim="#7f8c8d", fg_bright="#1a1a2e", fg_terminal="#d4d4d4",
    header_bg="#1a1a2e", header_fg="#ffffff", header_sub="#7f8fa6",
    border="#dcdde1", separator="#bdc3c7",
    accent="#2980b9", accent_hover="#3498db",
    success="#27ae60", success_fg="#ffffff",
    danger="#e74c3c", warning="#f39c12",
    purple="#8e44ad", teal="#16a085", orange="#d35400",
    src_mmt9="#2980b9", src_sdlcd="#e67e22", src_both="#27ae60",
    list_bg="#ffffff", list_fg="#2c3e50",
    list_select="#d5e8f0", list_select_fg="#2c3e50",
    plot_grid="#ecf0f1", plot_spine="#bdc3c7",
    plot_text="#2c3e50", plot_text_dim="#7f8c8d",
    plot_series=["#3498db","#e74c3c","#2ecc71","#f39c12",
                 "#9b59b6","#1abc9c","#e67e22","#34495e"],
)
_DARK = dict(
    bg="#1e1e2e", bg_card="#272740", bg_input="#1b1b30",
    bg_terminal="#11111b", bg_plot="#181825", bg_plot_ax="#181825",
    fg="#cdd6f4", fg_dim="#6c7086", fg_bright="#ffffff", fg_terminal="#cdd6f4",
    header_bg="#11111b", header_fg="#cdd6f4", header_sub="#6c7086",
    border="#45475a", separator="#45475a",
    accent="#89b4fa", accent_hover="#b4d0fb",
    success="#a6e3a1", success_fg="#1e1e2e",
    danger="#f38ba8", warning="#f9e2af",
    purple="#cba6f7", teal="#94e2d5", orange="#fab387",
    src_mmt9="#89b4fa", src_sdlcd="#fab387", src_both="#a6e3a1",
    list_bg="#1b1b30", list_fg="#cdd6f4",
    list_select="#cba6f7", list_select_fg="#1e1e2e",
    plot_grid="#45475a", plot_spine="#45475a",
    plot_text="#cdd6f4", plot_text_dim="#6c7086",
    plot_series=["#89b4fa","#f38ba8","#a6e3a1","#f9e2af",
                 "#cba6f7","#94e2d5","#fab387","#b4befe"],
)

# --- Preferences --------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parent.parent / "config.json"
_prefs = {}          # loaded once at startup

def _load_prefs():
    global _prefs
    try:
        if _CFG_PATH.exists():
            _prefs = json.loads(_CFG_PATH.read_text("utf-8"))
    except Exception:
        _prefs = {}

def save_prefs():
    """Write current preferences to disk."""
    try:
        _CFG_PATH.write_text(json.dumps(_prefs, indent=2), "utf-8")
    except Exception:
        pass

def pref(key, default=None):
    """Read a preference value."""
    return _prefs.get(key, default)

def set_pref(key, value):
    """Set a preference value (call save_prefs() later to persist)."""
    _prefs[key] = value

# --- Active palette (set once at startup) -------------------------------------
_load_prefs()
_current = pref("theme", "light")
if _current not in ("light", "dark"):
    _current = "light"

def T():
    """Return the active theme palette dict."""
    return _LIGHT if _current == "light" else _DARK

def is_dark():
    return _current == "dark"

def set_theme(mode):
    """Set theme for NEXT launch (saves pref, does not recolour widgets)."""
    global _current
    _current = mode
    set_pref("theme", mode)
    save_prefs()

# --- TTK style (called once at startup) --------------------------------------
def init_theme(root):
    t = T()
    style = ttk.Style()
    try: style.theme_use("clam")
    except Exception: pass

    style.configure("TNotebook", background=t["bg"], borderwidth=0)
    style.configure("TNotebook.Tab", font=(FONT,10,"bold"), padding=[14,5],
                    background=t["bg_card"], foreground=t["fg"])
    style.map("TNotebook.Tab",
              background=[("selected",t["accent"]),("active",t["border"])],
              foreground=[("selected","#ffffff"),("active",t["fg_bright"])])
    style.configure("Treeview", background=t["list_bg"], foreground=t["list_fg"],
                    fieldbackground=t["list_bg"], font=FONT_BODY, rowheight=22)
    style.configure("Treeview.Heading", font=FONT_SMALL_B,
                    background=t["bg_card"], foreground=t["fg_bright"])
    style.map("Treeview",
              background=[("selected",t["list_select"])],
              foreground=[("selected",t["list_select_fg"])])
    style.configure("TProgressbar", troughcolor=t["border"], background=t["accent"])
    style.configure("TCombobox", fieldbackground=t["bg_input"],
                    background=t["bg_card"], foreground=t["fg"],
                    selectbackground=t["list_select"],
                    selectforeground=t["list_select_fg"])
    style.map("TCombobox",
              fieldbackground=[("readonly",t["bg_input"])],
              selectbackground=[("readonly",t["list_select"])])
    style.configure("TSeparator", background=t["separator"])
    style.configure("TScrollbar", troughcolor=t["bg"], background=t["border"])
    # Force option-menu / popdown colours for combobox dropdowns
    root.option_add("*TCombobox*Listbox.background", t["bg_input"])
    root.option_add("*TCombobox*Listbox.foreground", t["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", t["list_select"])
    root.option_add("*TCombobox*Listbox.selectForeground", t["list_select_fg"])

# --- Widget factories (thin wrappers) ----------------------------------------
# These read T() once at call time — no stored refs needed.

def _bg(kw, key="bg"):
    kw.setdefault("bg", T()[key]); return kw

def styled_frame(parent, **kw):
    return tk.Frame(parent, **_bg(kw))

def card_frame(parent, **kw):
    t=T(); kw.setdefault("bg",t["bg_card"])
    kw.setdefault("highlightbackground",t["border"])
    kw.setdefault("highlightthickness",1)
    return tk.Frame(parent, **kw)

def styled_label(parent, text="", heading=False, dim=False, **kw):
    t=T(); bg=kw.pop("parent_bg",t["bg"]); kw.setdefault("bg",bg)
    if heading:   kw.setdefault("font",FONT_HEADING); kw.setdefault("fg",t["fg_bright"])
    elif dim:     kw.setdefault("font",FONT_SMALL);   kw.setdefault("fg",t["fg_dim"])
    else:         kw.setdefault("font",FONT_BODY);    kw.setdefault("fg",t["fg"])
    return tk.Label(parent, text=text, **kw)

def styled_entry(parent, textvariable=None, **kw):
    t=T(); kw.setdefault("bg",t["bg_input"]); kw.setdefault("fg",t["fg"])
    kw.setdefault("insertbackground",t["fg"]); kw.setdefault("font",FONT_BODY)
    kw.setdefault("relief","flat"); kw.setdefault("highlightthickness",1)
    kw.setdefault("highlightbackground",t["border"])
    kw.setdefault("highlightcolor",t["accent"])
    if textvariable: kw["textvariable"]=textvariable
    return tk.Entry(parent, **kw)

def styled_button(parent, text="", command=None, style="primary", **kw):
    t=T(); kw.setdefault("font",FONT_SMALL_B); kw.setdefault("cursor","hand2")
    kw.setdefault("relief","flat"); kw.setdefault("bd",0)
    if style=="primary":
        kw.setdefault("bg",t["accent"]); kw.setdefault("fg","#ffffff")
        kw.setdefault("activebackground",t["accent_hover"]); kw.setdefault("activeforeground","#ffffff")
    elif style=="success":
        kw.setdefault("bg",t["success"]); kw.setdefault("fg",t["success_fg"])
        kw.setdefault("activebackground",t["success"])
    elif style=="danger":
        kw.setdefault("bg",t["danger"]); kw.setdefault("fg","#ffffff")
        kw.setdefault("activebackground",t["danger"])
    else:  # flat
        kw.setdefault("bg",t["bg_card"]); kw.setdefault("fg",t["fg"])
        kw.setdefault("activebackground",t["border"])
    return tk.Button(parent, text=text, command=command, **kw)

def styled_listbox(parent, **kw):
    t=T(); kw.setdefault("font",FONT_MONO_SM)
    kw.setdefault("bg",t["list_bg"]); kw.setdefault("fg",t["list_fg"])
    kw.setdefault("selectbackground",t["list_select"])
    kw.setdefault("selectforeground",t["list_select_fg"])
    kw.setdefault("activestyle","none"); kw.setdefault("highlightthickness",0)
    kw.setdefault("relief","flat"); kw.setdefault("bd",0)
    return tk.Listbox(parent, **kw)

def styled_text(parent, terminal=False, **kw):
    t=T()
    if terminal: kw.setdefault("bg",t["bg_terminal"]); kw.setdefault("fg",t["fg_terminal"])
    else:        kw.setdefault("bg",t["bg_input"]);    kw.setdefault("fg",t["fg"])
    kw.setdefault("font",FONT_MONO_SM); kw.setdefault("wrap","word")
    kw.setdefault("relief","flat"); kw.setdefault("highlightthickness",1)
    kw.setdefault("highlightbackground",t["border"])
    return tk.Text(parent, **kw)

def styled_cb(parent, text="", variable=None, command=None, **kw):
    """Themed Checkbutton."""
    t=T(); bg=kw.pop("parent_bg",t["bg"]); kw.setdefault("bg",bg)
    kw.setdefault("fg",t["fg"]); kw.setdefault("selectcolor",t["bg_input"])
    kw.setdefault("activebackground",bg); kw.setdefault("activeforeground",t["fg_bright"])
    kw.setdefault("font",FONT_SMALL)
    return tk.Checkbutton(parent, text=text, variable=variable, command=command, **kw)

def styled_rb(parent, text="", variable=None, value=None, command=None, **kw):
    """Themed Radiobutton."""
    t=T(); bg=kw.pop("parent_bg",t["bg"]); kw.setdefault("bg",bg)
    kw.setdefault("fg",t["fg"]); kw.setdefault("selectcolor",t["bg_input"])
    kw.setdefault("activebackground",bg); kw.setdefault("activeforeground",t["fg_bright"])
    kw.setdefault("font",FONT_SMALL)
    return tk.Radiobutton(parent, text=text, variable=variable, value=value,
                          command=command, **kw)

def styled_lf(parent, text="", **kw):
    """Themed LabelFrame."""
    t=T(); kw.setdefault("bg",t["bg"]); kw.setdefault("fg",t["fg_bright"])
    kw.setdefault("font",FONT_SMALL_B); kw.setdefault("padx",8); kw.setdefault("pady",6)
    return tk.LabelFrame(parent, text=text, **kw)

# --- Matplotlib helpers -------------------------------------------------------
def style_figure(fig):
    fig.set_facecolor(T()["bg_plot"])

def style_axes(ax, title="", xlabel="", ylabel=""):
    t=T(); ax.set_facecolor(t["bg_plot_ax"])
    ax.grid(True, alpha=0.25, color=t["plot_grid"])
    ax.tick_params(colors=t["plot_text_dim"], labelsize=8)
    for s in ax.spines.values(): s.set_color(t["plot_spine"])
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", color=t["plot_text"])
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=t["plot_text_dim"])
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=t["plot_text_dim"])

def style_axes_empty(ax, msg="Select an item to view"):
    t=T(); ax.set_facecolor(t["bg_plot_ax"])
    ax.text(0.5,0.5,msg,ha="center",va="center",fontsize=13,color=t["fg_dim"],transform=ax.transAxes)
    ax.tick_params(colors=t["plot_text_dim"],labelsize=8)
    for s in ax.spines.values(): s.set_color(t["plot_spine"])
    ax.grid(True, alpha=0.1, color=t["plot_grid"])

def style_legend(ax, **kw):
    t=T(); kw.setdefault("fontsize",8); kw.setdefault("framealpha",0.6)
    kw.setdefault("edgecolor",t["plot_spine"]); kw.setdefault("facecolor",t["bg_plot_ax"])
    kw.setdefault("labelcolor",t["plot_text"]); return ax.legend(**kw)

def plot_color(i=0):
    c=T()["plot_series"]; return c[i%len(c)]