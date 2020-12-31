import asyncio
async def sum(a,b):
    # await asyncio.sleep(2)
    return a+b

async def main():
    l = []
    l.append(loop.create_task(sum(121212,123)))
    l.append(loop.create_task(sum(123,45)))
    print("hi")
    l.append(loop.create_task(sum(1,1)))
    await asyncio.wait(l)
    return l

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    s1,s2,s3 = loop.run_until_complete(main())
    print(s1,s2,s3)
    print(s3.result())
    loop.close()
