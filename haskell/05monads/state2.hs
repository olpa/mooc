-- https://stepik.org/lesson/8444/step/7?unit=1579

import Control.Monad.Writer
import Control.Monad.State

writerToState :: Monoid w => Writer w a -> State w a
writerToState m = state $ \st ->
  let (a, w) = runWriter m
  in (a, mappend st w)


t1 = runState (writerToState $ tell "world") "hello," -- ((),"hello,world")
t2 = runState (writerToState $ tell "world") mempty --  ((),"world")
