import asyncio
import websockets

class WsConnection:
    def __init__(self, websocket):
        pass

async def server(websocket, path):
    async for message in websocket:
        await websocket.send(message)

asyncio.get_event_loop().run_until_complete(
    websockets.serve(server, 'localhost', 8765))
asyncio.get_event_loop().run_forever()
