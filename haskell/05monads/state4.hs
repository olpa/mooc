-- https://stepik.org/lesson/8444/step/10?unit=1579

import Control.Monad.State

data Tree a = Leaf a | Fork (Tree a) a (Tree a) deriving Show

nTiM :: Tree () -> Integer -> State Integer (Tree Integer)
nTiM (Leaf ()) n = do
  put (n + 1)
  return (Leaf n)
nTiM (Fork l () r) n = do
  l' <- (nTiM l n)
  myN <- get
  r' <- (nTiM r (myN + 1))
  return (Fork l' myN r')

t1 = (runState $ (nTiM (Leaf ()) 4)) 777 -- (Leaf 4,5)
t2 = (runState $ (nTiM (Fork (Leaf ()) () (Leaf ())) 14)) 777

numberTree :: Tree () -> Tree Integer
numberTree tree = (evalState $ (nTiM tree 1)) 777

t11 = numberTree (Leaf ()) -- Leaf 1
t12 = numberTree (Fork (Leaf ()) () (Leaf ())) -- Fork (Leaf 1) 2 (Leaf 3)
