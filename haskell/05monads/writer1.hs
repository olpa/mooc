import Control.Monad.Writer

evalWriter :: Writer w a -> a
evalWriter m = val where (val, _) = runWriter m

t = evalWriter (return 3 :: Writer String Int)
