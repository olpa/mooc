-- https://stepik.org/lesson/8442/step/6?unit=1577
import Control.Monad.Writer

type Shopping = Writer (Sum Integer) ()

purchase :: String -> Integer -> Shopping
purchase item cost =  tell $ Sum cost

total :: Shopping -> Integer
total = getSum . execWriter


shopping1 :: Shopping
shopping1 = do
  purchase "Jeans"   19200
  purchase "Water"     180
  purchase "Lettuce"   328

t = total shopping1 -- 19708
