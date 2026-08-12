"""
Satellite Explorer
===================
A multi-tab GUI application for merging, exploring, and visualising
satellite catalogue data from SATCAT, UCS, DISCOS, and MMT-9 sources.

Usage:
    pip install pandas numpy matplotlib openpyxl
    python main.py
"""

import tkinter as tk
from gui.app import SatelliteExplorerApp


def main():
    root = tk.Tk()
    root.title("Satellite Explorer")
    root.geometry("1400x850")
    root.minsize(1100, 700)
    app = SatelliteExplorerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()