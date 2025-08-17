import asyncio
import aiohttp
import time
from typing import List, Dict, Any
import json

class E2ETestRunner:
    def __init__(self, proxy_url: str = "http://localhost:8080", inference_url: str = "http://localhost:8081"):
        self.proxy_url = proxy_url
        self.inference_url = inference_url
    
    async def make_embed_request(self, session: aiohttp.ClientSession, url: str, inputs: List[str]) -> Dict[str, Any]:
        """Make a single embed request to the specified URL."""
        payload = {"inputs": inputs}
        async with session.post(f"{url}/embed", json=payload) as response:
            if response.status != 200:
                raise Exception(f"Request failed with status {response.status}: {await response.text()}")
            return await response.json()
    
    async def test_single_request(self, session: aiohttp.ClientSession, url: str, test_name: str) -> float:
        """Test a single request and measure time."""
        print(f"\n=== {test_name} ===")
        inputs = ["Hello, world!"]
        
        start_time = time.time()
        result = await self.make_embed_request(session, url, inputs)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"Single request took: {elapsed:.2f} seconds")
        print(f"Response: {len(result['embeddings'])} embeddings")
        
        return elapsed
    
    async def test_concurrent_requests(self, session: aiohttp.ClientSession, url: str, test_name: str, num_requests: int = 10) -> float:
        """Test multiple concurrent requests and measure total time."""
        print(f"\n=== {test_name} ===")
        inputs_list = [f"Request {i}: Hello from concurrent test!" for i in range(num_requests)]
        
        start_time = time.time()
        
        # Create tasks for concurrent requests
        tasks = []
        for i, input_text in enumerate(inputs_list):
            task = self.make_embed_request(session, url, [input_text])
            tasks.append(task)
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"{num_requests} concurrent requests took: {elapsed:.2f} seconds")
        print(f"Average per request: {elapsed/num_requests:.2f} seconds")
        print(f"Total embeddings received: {sum(len(r['embeddings']) for r in results)}")
        
        return elapsed
    
    async def test_sequential_requests(self, session: aiohttp.ClientSession, url: str, test_name: str, num_requests: int = 10) -> float:
        """Test multiple sequential requests and measure total time."""
        print(f"\n=== {test_name} ===")
        
        start_time = time.time()
        
        for i in range(num_requests):
            input_text = f"Sequential request {i}: Hello from sequential test!"
            await self.make_embed_request(session, url, [input_text])
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"{num_requests} sequential requests took: {elapsed:.2f} seconds")
        print(f"Average per request: {elapsed/num_requests:.2f} seconds")
        
        return elapsed
    
    async def run_performance_comparison(self):
        """Run complete performance comparison tests."""
        print("Starting E2E Performance Tests")
        print("=" * 50)
        
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                # Test 1: Single request to proxy
                single_proxy_time = await self.test_single_request(
                    session, self.proxy_url, "Single Request via Proxy"
                )
                
                # Test 2: 10 concurrent requests to proxy
                concurrent_proxy_time = await self.test_concurrent_requests(
                    session, self.proxy_url, "10 Concurrent Requests via Proxy", 10
                )
                
                # Test 3: 10 sequential requests to proxy (baseline)
                sequential_proxy_time = await self.test_sequential_requests(
                    session, self.proxy_url, "10 Sequential Requests via Proxy", 10
                )
                
                # Test 4: Direct inference service tests for comparison
                print(f"\n=== Direct Inference Service Tests ===")
                single_inference_time = await self.test_single_request(
                    session, self.inference_url, "Single Request to Inference Service"
                )
                
                concurrent_inference_time = await self.test_concurrent_requests(
                    session, self.inference_url, "10 Concurrent Requests to Inference Service", 10
                )
                
                # Performance Analysis
                print(f"\n{'='*60}")
                print("PERFORMANCE ANALYSIS")
                print(f"{'='*60}")
                
                print(f"Single request via proxy:        {single_proxy_time:.2f}s")
                print(f"10 concurrent via proxy:         {concurrent_proxy_time:.2f}s")
                print(f"10 sequential via proxy:         {sequential_proxy_time:.2f}s")
                print(f"Single request to inference:     {single_inference_time:.2f}s")
                print(f"10 concurrent to inference:      {concurrent_inference_time:.2f}s")
                
                # Expected behavior analysis
                print(f"\n{'='*60}")
                print("EXPECTED BEHAVIOR ANALYSIS")
                print(f"{'='*60}")
                
                # Current proxy is just forwarding, so concurrent should be similar to direct inference
                concurrent_ratio = concurrent_proxy_time / single_proxy_time
                print(f"Concurrent/Single ratio (proxy): {concurrent_ratio:.2f}x")
                
                # Check if concurrent requests are taking approximately 10x longer 
                # (indicating proper sequential processing in the inference service)
                if 8.0 <= concurrent_ratio <= 12.0:
                    print("✅ PASS: Concurrent requests are taking ~10x longer as expected")
                    print("   This indicates proper sequential processing in the inference service")
                else:
                    print("❌ FAIL: Concurrent requests timing is unexpected")
                    print(f"   Expected ~10x longer, got {concurrent_ratio:.1f}x longer")
                
                # Future expectations (when batching is implemented)
                print(f"\nFUTURE EXPECTATIONS (with batching):")
                print(f"- 10 concurrent requests should take ~1-2 seconds (close to single request time)")
                print(f"- Current time: {concurrent_proxy_time:.2f}s")
                print(f"- Target time: ~{single_proxy_time:.2f}s")
                print(f"- Potential improvement: {concurrent_proxy_time/single_proxy_time:.1f}x faster")
                
            except Exception as e:
                print(f"Test failed with error: {e}")
                raise

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run E2E tests for auto-batching proxy")
    parser.add_argument("--proxy-url", default="http://localhost:8080", 
                       help="URL of the proxy service (default: http://localhost:8080)")
    parser.add_argument("--inference-url", default="http://localhost:8081", 
                       help="URL of the inference service (default: http://localhost:8081)")
    
    args = parser.parse_args()
    
    runner = E2ETestRunner(args.proxy_url, args.inference_url)
    await runner.run_performance_comparison()

if __name__ == "__main__":
    asyncio.run(main())