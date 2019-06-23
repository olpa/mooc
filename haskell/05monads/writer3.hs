-- https://stepik.org/lesson/8442/step/7?unit=1577
import Control.Monad.Writer

type Shopping = Writer (Sum Integer, [String]) ()

purchase :: String -> Integer -> Shopping
purchase item cost =  tell $ (Sum cost, [item])

total :: Shopping -> Integer
total = getSum . fst . execWriter

items :: Shopping -> [String]
items = snd . execWriter


shopping1 :: Shopping
shopping1 = do
  purchase "Jeans"   19200
  purchase "Water"     180
  purchase "Lettuce"   328

t1 = total shopping1 -- 19708
t2 = items shopping1 -- ["Jeans", "Water", "Lettuce"]
