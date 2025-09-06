import logging

from colorama import Back, Fore, Style

_lf = f"{Back.LIGHTBLACK_EX}"
_rg = f"{Back.RESET}{Fore.RESET}"
_lvlname = {
    "INFO": f"     {_lf} INFO {_rg}",
    "DEBUG": f"    {_lf} DEBUG {_rg}",
    "WARNING": f"  {_lf} WARNING {_rg}",
    "ERROR": f"    {_lf} ERROR {_rg}",
}


class CustomFormatter(logging.Formatter):
    def format(self, record):
        name = record.name[(8 if record.name.startswith("pynguin") else 0) :]

        log_format = (
            f"{_lvlname[record.levelname]}  "
            f"{Fore.CYAN}"
            f"{name:<15} \033[1;37m| "
            f"{Style.RESET_ALL}"
            "{message}"
        )

        formatter = logging.Formatter(log_format, style="{")
        return formatter.format(record)