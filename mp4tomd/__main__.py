"""包入口：有参数走 CLI，无参数启动 GUI。"""
import sys

__version__ = "0.1.0"


def main():
    if len(sys.argv) > 1:
        from . import cli
        raise SystemExit(cli.main())
    from . import gui
    gui.main()


if __name__ == "__main__":
    main()
