import asyncio


def foo():
    async def bar():
        return 42

    return bar()


async def main():
    print(await foo())


asyncio.run(main())
