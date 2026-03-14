# Building a Python-requests-style HTTP Client in Haskell

## The Problem with Learning Haskell

When I started learning Haskell, I always struggled to find a suitable scenario to apply it. Without a concrete problem to solve, it is difficult to make real progress.

At some point I had some practical work to do: fetch data from HTTP APIs, process the results, and post them to other endpoints. My instinct was to reach for Python and the `requests` library, because it is genuinely handy for this kind of task. You grab a URL, get back some JSON, transform it, and send it somewhere else. The whole thing can be done in a few lines.

But I thought: this is exactly the kind of real problem I need. Why not try Haskell?

## Exploring the Existing Ecosystem

The first thing I reached for was `http-client`. It is powerful and the foundation for most other Haskell HTTP libraries. But using it directly requires quite a bit of setup for simple tasks. You need to create a manager, parse a request, and work with a set of low-level types. It is a great library, but it is designed for cases where you need precise control.

There are some lighter alternatives. `wreq` is a popular one, built on a lens API. Lenses are a fascinating part of the Haskell ecosystem and many people find them worth learning. `req` is another option with a clean, type-safe API. I used `req` to finish the work I was doing, and it worked well. But I still missed the simplicity of Python's `requests`, and I kept thinking about whether something like it was possible in Haskell.

## Finding http-dispatch and Deciding to Build Something

Some time later, I came across `http-dispatch`. It was much closer to what I had in mind. The API was straightforward and the concepts were familiar. Unfortunately the library had not been maintained for a long time. There were some bugs, and some fixes in the source code had never made it to Hackage.

That is when I decided to build my own.

The goal was simple: a library for scenarios where you do not need fine-grained control over the HTTP lifecycle. For those cases, `http-client` is the right tool. But for the common case of making a request and getting back a response, I wanted something that felt closer to the Python experience.

## The Core Abstraction

I started from the most basic mental model of an HTTP interaction: you have a request, you send it to a server, and you get back a response.

This became the core API:

```haskell
data Request a = Request
  { method  :: Method
  , url     :: String
  , headers :: Headers
  , body    :: Maybe a
  }

data Response a = Response
  { status  :: Int
  , headers :: Headers
  , body    :: a
  }

send :: (ToRequestBody a, FromResponseBody b) => Request a -> IO (Response b)
```

The `send` function takes a `Request` and produces a `Response` in `IO`. The type parameters carry the body types, and the typeclass constraints handle serialization and deserialization automatically.

For common HTTP methods, there are simple shortcuts:

```haskell
-- A simple GET request
resp <- get "https://example.com" :: IO (Response String)
print resp.status  -- 200
print resp.body    -- HTML content as String

-- POST with a plain text body
resp <- post "https://httpbin.org/post" ("hello" :: String) :: IO (Response String)
print resp.status  -- 200
```

You can also construct a `Request` manually when you need custom headers:

```haskell
let req = Request
      { method  = GET
      , url     = "https://api.example.com/data"
      , headers = [("Authorization", "Bearer my-token")]
      , body    = Nothing :: Maybe BS.ByteString
      }
resp <- send req :: IO (Response String)
```

The library supports modern Haskell record dot syntax (`resp.status`, `resp.body`) as well as traditional accessor functions (`responseStatus resp`, `responseBody resp`) for those who prefer not to enable language extensions.

## JSON Integration

The feature I most wanted to replicate from Python's `requests` is its JSON handling. In Python, you pass `json=` to your request and call `.json()` on the response. It handles the content type header, the serialization, and the deserialization for you. Most API work is exactly this: send JSON, receive JSON.

Haskell's type system makes this possible in a way that is, I think, even nicer than Python. Because we declare our data types upfront, we get static guarantees about the shape of the data.

Here is an example of parsing a JSON response:

```haskell
{-# LANGUAGE DeriveGeneric #-}

import Data.Aeson (FromJSON)
import GHC.Generics (Generic)
import Network.HTTP.Request

data Date = Date
  { __type :: String
  , iso     :: String
  } deriving (Show, Generic)

instance FromJSON Date

main :: IO ()
main = do
  response <- get "https://api.leancloud.cn/1.1/date" :: IO (Response Date)
  print response.status  -- 200
  print response.body    -- Date { __type = "Date", iso = "2026-03-14T..." }
```

There is no explicit parsing step. The type annotation `:: IO (Response Date)` is enough to tell the library to decode the JSON body into a `Date` value. If decoding fails, an `AesonException` is thrown.

Sending JSON works the same way:

```haskell
data Message = Message { content :: String } deriving (Generic)

instance ToJSON Message

main :: IO ()
main = do
  resp <- post "https://api.example.com/messages" (Message "Hello")
            :: IO (Response String)
  print resp.status
```

Any type with a `ToJSON` instance is automatically serialized, and `Content-Type: application/json` is set on the request. You do not need to think about it.

This gives you the simplicity of Python's `requests` combined with the compile-time guarantees of a static type system. You define your API shapes as types, and the compiler helps you use them correctly.

## Using It for an LLM Agent

With the library in a working state, I wanted to build something more substantial with it. I decided to implement a small LLM agent. That became `hasuke`, a CLI tool for interacting with Claude.

Calling an Anthropic-style API is exactly the use case this library was built for. You construct a JSON request body, send it to the endpoint, and get back a JSON response. The library handled all of this without any friction.

But after building the first version, I noticed that it would sit silently until the full response was generated, then display everything at once. Modern LLM providers support streaming responses to address this: they send partial results incrementally using Server-Sent Events, so you start seeing output right away.

## Adding Streaming and SSE Support

I extended the library to support streaming. The key was to express this within the existing type system without changing the core API. I introduced a `StreamBody` type:

```haskell
data StreamBody a = StreamBody
  { readNext    :: IO (Maybe a)
  , closeStream :: IO ()
  }
```

To receive a streaming response, you just change the type annotation:

```haskell
let req = Request GET "https://example.com/stream" [] (Nothing :: Maybe BS.ByteString)
resp <- send req :: IO (Response (StreamBody BS.ByteString))

let loop = do
      mChunk <- resp.body.readNext
      case mChunk of
        Nothing    -> return ()
        Just chunk -> BS.putStr chunk >> loop
loop
resp.body.closeStream
```

For SSE, the library parses the event stream protocol automatically. Each `SseEvent` has fields for the data, event type, and event id:

```haskell
data SseEvent = SseEvent
  { sseData :: T.Text
  , sseType :: Maybe T.Text
  , sseId   :: Maybe T.Text
  }
```

Using it looks like this:

```haskell
let req = Request POST "https://api.anthropic.com/v1/messages" headers (Just body)
resp <- send req :: IO (Response (StreamBody SseEvent))

let loop = do
      mEvent <- resp.body.readNext
      case mEvent of
        Nothing    -> return ()
        Just event -> T.putStr event.sseData >> loop
loop
resp.body.closeStream
```

The `send` function signature did not change at all. When the target type is `StreamBody SseEvent`, the library keeps the connection open and streams events through an internal buffer. From the caller's side, you are just getting a different kind of response body.

This is where Haskell's type system earns its keep. Adding a completely different data transfer mode required almost no changes to the existing API. The same `send` function, the same `Request` type, the same conventions. It is now working well in `hasuke`.

## Where Things Stand

The library is published on Hackage under the name `request` and can be installed with cabal or stack in the usual way. The source code is on GitHub at https://github.com/aisk/request. It powers the streaming output in `hasuke`, which you can find at https://github.com/aisk/hasuke.

There are features still missing, such as support for HTML form encoding and some other less common use cases. I plan to add these gradually over time. If you run into something you need, feel free to open an issue on GitHub. Feedback and contributions are very welcome.

For the common case of calling JSON APIs, whether in a single response or as a stream, the library does what it was built to do. If you are learning Haskell and looking for a practical project, or you just need a lightweight HTTP client, it might be worth a try.
