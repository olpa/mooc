-- https://stepik.org/lesson/8444/step/9?unit=1579

import Control.Monad
import Control.Monad.State

fibStep :: State (Integer, Integer) ()
fibStep = state $ \(a, b) -> ((), (b, a + b))

execStateN :: Int -> State s a -> s -> s
execStateN n m = execState $ replicateM_ n m

t1 = execState fibStep (0,1) -- (1,1)
t2 = execState fibStep (1,1) -- (1,2)
t3 = execState fibStep (1,2) --  (2,3)

fib :: Int -> Integer
fib n = fst $ execStateN n fibStep (0, 1)

t4 = map fib [1..20]
