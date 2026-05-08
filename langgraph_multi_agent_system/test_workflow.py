#!/usr/bin/env python3
"""
Comprehensive test script for LangGraph multi-agent workflow.

This script tests the complete workflow with a complex input that requires
all agents (equity, tax, and risk) to be invoked and their results compiled.
"""

import os
import sys
import asyncio
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), 'src', '.env'))  

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.orchestrator_langgraph import LangGraphNativeOrchestrator
from src.models.request_models import FinancialRequest, WorkflowType, AnalysisType, RiskTolerance


class WorkflowTester:
    """Test class for comprehensive workflow validation."""
    
    def __init__(self):
        self.orchestrator = LangGraphNativeOrchestrator()
        self.test_results = {}
        
    async def test_comprehensive_workflow(self):
        """Test workflow with input requiring all agents."""
        
        print("🚀 Starting Comprehensive Workflow Test")
        print("=" * 60)
        
        # Test input that should trigger all agents
        test_input = "Get the today's stock price for TSLA and AAPL. Are there any risks to sell those in 3 months if I purchase with the current market price? and recommend any tax savings."
        
        print(f"📝 Test Input: {test_input}")
        print()
        
        try:
            # Execute workflow with timeout
            print("⚡ Executing workflow...")
            start_time = datetime.now()
            
            try:
                # Parse the test input to extract components
                # The input contains stock analysis, risk assessment, and tax consultation
                # This should trigger COMPREHENSIVE_ANALYSIS workflow
                result = await asyncio.wait_for(
                    self.orchestrator.process_financial_request(
                        user_id="test_user_001",
                        stocks=[
                            {"symbol": "TSLA", "analysis_type": "technical"},
                            {"symbol": "AAPL", "analysis_type": "technical"}
                        ],
                        tax_question="recommend any tax savings",
                        portfolio_value=100000.0,  # Estimated portfolio value
                        risk_tolerance="moderate",
                        time_horizon=3,  # 3 months
                        current_holdings=["TSLA", "AAPL"],
                        request_id="test_req_001"
                    ),
                    timeout=120.0  # 2 minute total timeout for entire workflow
                )
            except asyncio.TimeoutError:
                print("❌ Workflow timed out after 2 minutes")
                return
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            print(f"✅ Workflow completed in {execution_time:.2f} seconds")
            print()
            
            # Analyze results
            await self._analyze_results(result, execution_time)
            
        except Exception as e:
            print(f"❌ Workflow failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    async def _analyze_results(self, result: Dict[str, Any], execution_time: float):
        """Analyze workflow execution results."""
        
        print("📊 Workflow Results Analysis")
        print("-" * 40)
        
        # Check workflow completion
        status = result.get('status', 'unknown')
        print(f"📋 Workflow Status: {status}")
        
        if status == 'completed':
            print("✅ Workflow completed successfully")
        else:
            print(f"⚠️  Workflow ended with status: {status}")
        
        # Check which agents were invoked
        agent_results = result.get('agent_results', [])
        print(f"\n🤖 Agents Invoked: {len(agent_results)}")
        
        expected_agents = ['equity_expert', 'tax_expert', 'risk_expert']
        
        # Handle both list and dict formats
        if isinstance(agent_results, dict):
            invoked_agents = list(agent_results.keys())
        elif isinstance(agent_results, list):
            invoked_agents = [result.get('agent_name', 'unknown') for result in agent_results]
        else:
            invoked_agents = []
        
        print(f"Expected agents: {expected_agents}")
        print(f"Actually invoked: {invoked_agents}")
        
        # Verify each agent was called
        for agent_name in expected_agents:
            if isinstance(agent_results, dict) and agent_name in agent_results:
                agent_result = agent_results[agent_name]
                print(f"\n✅ {agent_name.upper()} Agent:")
                print(f"   Status: {agent_result.get('status', 'unknown')}")
                
                # Check if agent produced meaningful output
                result_data = agent_result.get('result_data', {})
                if result_data:
                    response = result_data.get('response', '')
                    if response:
                        print(f"   Response length: {len(response)} characters")
                        print(f"   Response preview: {response[:200]}...")
                    else:
                        print("   ⚠️  No response content")
                else:
                    print("   ⚠️  No result data")
            elif isinstance(agent_results, list):
                # Find agent in list
                agent_found = False
                for result in agent_results:
                    if result.get('agent_name') == agent_name:
                        agent_found = True
                        print(f"\n✅ {agent_name.upper()} Agent:")
                        print(f"   Status: {result.get('status', 'unknown')}")
                        
                        # Check if agent produced meaningful output
                        result_data = result.get('result_data', {})
                        if result_data:
                            response = result_data.get('response', '')
                            if response:
                                print(f"   Response length: {len(response)} characters")
                                print(f"   Response preview: {response[:200]}...")
                            else:
                                print("   ⚠️  No response content")
                        else:
                            print("   ⚠️  No result data")
                        break
                
                if not agent_found:
                    print(f"\n❌ {agent_name.upper()} Agent: NOT INVOKED")
            else:
                print(f"\n❌ {agent_name.upper()} Agent: NOT INVOKED")
        
        # Check final recommendations
        final_results = result.get('final_results', {})
        if final_results:
            print(f"\n📋 Final Results:")
            print(f"   Workflow type: {final_results.get('workflow_type', 'unknown')}")
            print(f"   Agents executed: {final_results.get('agents_executed', [])}")
            print(f"   Execution time: {final_results.get('execution_time', 0):.2f}s")
            
            # Check for comprehensive analysis
            recommendations = final_results.get('final_recommendations', [])
            if recommendations:
                print(f"   Recommendations: {len(recommendations)} items")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"     {i}. {rec[:100]}...")
            else:
                print("   ⚠️  No final recommendations found")
        
        # Check for errors
        errors = result.get('errors', [])
        if errors:
            print(f"\n⚠️  Errors encountered: {len(errors)}")
            for error in errors:
                print(f"   - {error}")
        else:
            print(f"\n✅ No errors encountered")
        
        # Performance metrics
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Total execution time: {execution_time:.2f}s")
        print(f"   Agents processed: {len(agent_results)}")
        if len(agent_results) > 0:
            avg_time_per_agent = execution_time / len(agent_results)
            print(f"   Average time per agent: {avg_time_per_agent:.2f}s")
        
        # Test validation
        self._validate_test_results(expected_agents, invoked_agents, status, result)
    
    def _validate_test_results(self, expected_agents, invoked_agents, status, result):
        """Validate that test meets acceptance criteria."""
        
        print(f"\n🔍 Test Validation")
        print("-" * 20)
        
        validation_passed = True
        
        # Check 1: All expected agents were invoked
        missing_agents = set(expected_agents) - set(invoked_agents)
        if missing_agents:
            print(f"❌ Missing agents: {missing_agents}")
            validation_passed = False
        else:
            print("✅ All expected agents were invoked")
        
        # Check 2: Workflow completed successfully
        if status != 'completed':
            print(f"❌ Workflow did not complete successfully (status: {status})")
            validation_passed = False
        else:
            print("✅ Workflow completed successfully")
        
        # Check 3: Each agent produced results
        agent_results = result.get('agent_results', {})
        for agent in expected_agents:
            if agent in agent_results:
                agent_data = agent_results[agent]
                if not agent_data.get('result_data'):
                    print(f"❌ {agent} did not produce result data")
                    validation_passed = False
                else:
                    print(f"✅ {agent} produced result data")
        
        # Check 4: Final compilation exists
        if not result.get('final_results'):
            print("❌ No final results compilation")
            validation_passed = False
        else:
            print("✅ Final results compiled")
        
        # Overall result
        print(f"\n🏆 Test Result: {'PASSED' if validation_passed else 'FAILED'}")
        
        if validation_passed:
            print("🎉 All agents were successfully invoked and workflow completed!")
        else:
            print("⚠️  Some validation criteria were not met.")
        
        self.test_results['validation_passed'] = validation_passed
        self.test_results['expected_agents'] = expected_agents
        self.test_results['invoked_agents'] = invoked_agents
        self.test_results['status'] = status


async def main():
    """Main test execution."""
    
    print("🧪 LangGraph Multi-Agent Workflow Test")
    print("=" * 50)
    print("Testing comprehensive workflow with input requiring all agents...")
    print()
    
    tester = WorkflowTester()
    await tester.test_comprehensive_workflow()
    
    print(f"\n📋 Test Summary:")
    print(f"   Validation: {'PASSED' if tester.test_results.get('validation_passed') else 'FAILED'}")
    print(f"   Expected agents: {tester.test_results.get('expected_agents', [])}")
    print(f"   Invoked agents: {tester.test_results.get('invoked_agents', [])}")
    print(f"   Final status: {tester.test_results.get('status', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
