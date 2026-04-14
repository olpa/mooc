import asyncio
import aiohttp
import time
import argparse
from typing import List, Dict, Any, Tuple
import statistics


class E2ETestRunner:
    def __init__(self, proxy_url: str = "http://localhost:8080", inference_url: str = "http://localhost:8081"):
        self.proxy_url = proxy_url
        self.inference_url = inference_url
        
        # Test configuration variables
        self.BATCH_SIZE = 32
        self.WAIT_TIME_MS = 100
        self.INFERENCE_RESPONSE_TIME_S = 1.0
        
        # Test concurrency levels
        self.CONCURRENCY_LEVELS = [1, 3, 10, 100]

    async def make_embed_request(self, session: aiohttp.ClientSession, url: str, text: str) -> Tuple[Dict[str, Any], float]:
        """Make a single embed request and return response with elapsed time."""
        payload = {"inputs": [text]}
        
        start_time = time.time()
        async with session.post(f"{url}/embed", json=payload) as response:
            if response.status != 200:
                raise Exception(f"Request failed with status {response.status}: {await response.text()}")
            result = await response.json()
            elapsed_time = time.time() - start_time
            
        return result, elapsed_time

    async def run_concurrent_test(self, session: aiohttp.ClientSession, url: str, num_requests: int) -> Dict[str, Any]:
        """Run concurrent requests and measure timing."""
        print(f"Running {num_requests} concurrent requests...")
        
        # Create tasks for concurrent requests
        tasks = []
        for i in range(num_requests):
            text = f"Test request {i}"
            task = self.make_embed_request(session, url, text)
            tasks.append(task)
        
        # Execute all requests concurrently and measure total time
        overall_start = time.time()
        results = await asyncio.gather(*tasks)
        overall_elapsed = time.time() - overall_start
        
        # Extract individual times and responses
        individual_times = [elapsed for _, elapsed in results]
        responses = [response for response, _ in results]
        
        return {
            'num_requests': num_requests,
            'overall_time': overall_elapsed,
            'individual_times': individual_times,
            'avg_individual_time': statistics.mean(individual_times),
            'min_time': min(individual_times),
            'max_time': max(individual_times),
            'responses': responses
        }

    def validate_expectations(self, service_name: str, results: Dict[str, Any]) -> List[str]:
        """Validate timing expectations based on service type and configuration."""
        validations = []
        num_requests = results['num_requests']
        overall_time = results['overall_time']
        
        if service_name == "inference":
            # Direct inference service: requests are processed sequentially
            # Expected time: num_requests * INFERENCE_RESPONSE_TIME_S
            expected_time = num_requests * self.INFERENCE_RESPONSE_TIME_S
            tolerance = 0.2  # 20% tolerance
            
            if abs(overall_time - expected_time) <= expected_time * tolerance:
                validations.append(f"✅ PASS: Overall time {overall_time:.2f}s is within expected range {expected_time:.2f}s ±{tolerance*100}%")
            else:
                validations.append(f"❌ FAIL: Overall time {overall_time:.2f}s, expected ~{expected_time:.2f}s")
                
        elif service_name == "proxy":
            # Proxy service expectations depend on batching implementation
            if num_requests == 1:
                # Single request should be similar to inference service
                expected_time = self.INFERENCE_RESPONSE_TIME_S + (self.WAIT_TIME_MS / 1000)
                tolerance = 0.3
                
                if overall_time <= expected_time * (1 + tolerance):
                    validations.append(f"✅ PASS: Single request time {overall_time:.2f}s is reasonable")
                else:
                    validations.append(f"❌ FAIL: Single request time {overall_time:.2f}s is too high")
                    
            elif num_requests <= self.BATCH_SIZE:
                # Requests that fit in one batch
                # Expected: WAIT_TIME + INFERENCE_RESPONSE_TIME
                expected_time = (self.WAIT_TIME_MS / 1000) + self.INFERENCE_RESPONSE_TIME_S
                tolerance = 0.3
                
                if overall_time <= expected_time * (1 + tolerance):
                    validations.append(f"✅ PASS: Batch time {overall_time:.2f}s is efficient (≤{expected_time * (1 + tolerance):.2f}s)")
                else:
                    # Check if proxy is implemented vs just forwarding
                    sequential_time = num_requests * self.INFERENCE_RESPONSE_TIME_S
                    if abs(overall_time - sequential_time) <= sequential_time * 0.2:
                        validations.append(f"⚠️  INFO: Time {overall_time:.2f}s suggests proxy is forwarding sequentially (batching not yet implemented)")
                    else:
                        validations.append(f"❌ FAIL: Unexpected timing {overall_time:.2f}s for {num_requests} requests")
                        
            else:
                # Requests requiring multiple batches
                num_batches = (num_requests + self.BATCH_SIZE - 1) // self.BATCH_SIZE
                expected_time = num_batches * ((self.WAIT_TIME_MS / 1000) + self.INFERENCE_RESPONSE_TIME_S)
                tolerance = 0.3
                
                if overall_time <= expected_time * (1 + tolerance):
                    validations.append(f"✅ PASS: Multi-batch time {overall_time:.2f}s is efficient for {num_batches} batches")
                else:
                    validations.append(f"❌ FAIL: Multi-batch time {overall_time:.2f}s is inefficient (expected ~{expected_time:.2f}s)")
        
        return validations

    def report_results(self, service_name: str, results: Dict[str, Any]):
        """Print detailed results for a test run."""
        print(f"\n--- {service_name.upper()} SERVICE RESULTS ({results['num_requests']} concurrent requests) ---")
        print(f"Overall time: {results['overall_time']:.3f}s")
        print(f"Average per request: {results['avg_individual_time']:.3f}s")
        print(f"Min/Max individual times: {results['min_time']:.3f}s / {results['max_time']:.3f}s")
        print(f"Throughput: {results['num_requests'] / results['overall_time']:.1f} requests/second")
        
        # Validate expectations
        validations = self.validate_expectations(service_name, results)
        for validation in validations:
            print(validation)

    async def run_all_tests(self):
        """Run complete test suite against both services."""
        print("=" * 70)
        print("AUTO-BATCHING PROXY E2E PERFORMANCE TESTS")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Batch size: {self.BATCH_SIZE}")
        print(f"  Wait time: {self.WAIT_TIME_MS}ms")
        print(f"  Inference response time: {self.INFERENCE_RESPONSE_TIME_S}s")
        print(f"  Proxy URL: {self.proxy_url}")
        print(f"  Inference URL: {self.inference_url}")
        
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            all_results = {}
            
            for concurrency in self.CONCURRENCY_LEVELS:
                print(f"\n{'='*70}")
                print(f"TESTING WITH {concurrency} CONCURRENT REQUESTS")
                print(f"{'='*70}")
                
                # Test against inference service (direct)
                try:
                    inference_results = await self.run_concurrent_test(session, self.inference_url, concurrency)
                    self.report_results("inference", inference_results)
                    all_results[f"inference_{concurrency}"] = inference_results
                except Exception as e:
                    print(f"❌ FAIL: Inference service test failed: {e}")
                    continue
                
                print()
                
                # Test against proxy service
                try:
                    proxy_results = await self.run_concurrent_test(session, self.proxy_url, concurrency)
                    self.report_results("proxy", proxy_results)
                    all_results[f"proxy_{concurrency}"] = proxy_results
                except Exception as e:
                    print(f"❌ FAIL: Proxy service test failed: {e}")
                    continue
                
                # Compare services
                if f"inference_{concurrency}" in all_results and f"proxy_{concurrency}" in all_results:
                    inference_time = all_results[f"inference_{concurrency}"]["overall_time"]
                    proxy_time = all_results[f"proxy_{concurrency}"]["overall_time"]
                    
                    if proxy_time < inference_time * 0.8:  # 20% faster
                        speedup = inference_time / proxy_time
                        print(f"🚀 PROXY ADVANTAGE: {speedup:.1f}x faster than direct inference!")
                    elif proxy_time > inference_time * 1.2:  # 20% slower
                        slowdown = proxy_time / inference_time
                        print(f"⚠️  PROXY OVERHEAD: {slowdown:.1f}x slower than direct inference")
                    else:
                        print(f"📊 COMPARABLE: Proxy and inference times are similar")
            
            # Final summary
            print(f"\n{'='*70}")
            print("FINAL SUMMARY")
            print(f"{'='*70}")
            
            for concurrency in self.CONCURRENCY_LEVELS:
                if f"proxy_{concurrency}" in all_results:
                    proxy_time = all_results[f"proxy_{concurrency}"]["overall_time"]
                    print(f"{concurrency:3d} requests via proxy: {proxy_time:6.2f}s")
                if f"inference_{concurrency}" in all_results:
                    inference_time = all_results[f"inference_{concurrency}"]["overall_time"]
                    print(f"{concurrency:3d} requests via inference: {inference_time:6.2f}s")
                print()


async def main():
    parser = argparse.ArgumentParser(description="Run E2E tests for auto-batching proxy")
    parser.add_argument("--proxy-url", default="http://localhost:8080", 
                       help="URL of the proxy service (default: http://localhost:8080)")
    parser.add_argument("--inference-url", default="http://localhost:8081", 
                       help="URL of the inference service (default: http://localhost:8081)")
    
    args = parser.parse_args()
    
    runner = E2ETestRunner(args.proxy_url, args.inference_url)
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())