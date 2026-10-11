from .cli import main

if __name__ == "__main__":  # not in worker processes (Params.workers), which import this module
    main()
