-- https://stepik.org/lesson/8444/step/6?unit=1579

import Control.Monad.Reader
import Control.Monad.State

readerToState :: Reader r a -> State r a
readerToState m = state $ \st ->
  let a = runReader m st
  in (a, st)

t1 = evalState (readerToState $ asks (+2)) 4
t2 = runState (readerToState $ asks (+2)) 4
