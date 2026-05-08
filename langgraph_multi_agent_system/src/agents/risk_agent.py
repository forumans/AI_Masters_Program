"""Risk assessment agent for LangGraph multi-agent system."""

from typing import Dict, Any, List
from datetime import datetime
import math

from .base_agent import BaseLangGraphAgent, AgentConfig, agent_registry
from ..models.request_models import RiskRequest, RiskTolerance
from ..utils.monitoring import get_logger, metrics

logger = get_logger(__name__)


class RiskAgent(BaseLangGraphAgent):
    """Agent for portfolio risk assessment and management."""
    
    def __init__(self):
        config = AgentConfig(
            name="risk_expert",
            description="Assesses portfolio risk and provides risk management strategies",
            llm_model="gpt-3.5-turbo",
            max_tokens=1500,
            temperature=0.1,
            timeout=30.0,
            max_retries=3
        )
        super().__init__(config)
        agent_registry.register(self)
    
    def get_system_prompt(self) -> str:
        """Get system prompt for risk assessment."""
        return """You are a risk management specialist with expertise in portfolio risk analysis, asset allocation, and risk mitigation strategies.

Your responsibilities:
1. Assess portfolio risk based on holdings and market conditions
2. Calculate risk metrics (VaR, volatility, beta, etc.)
3. Provide risk-adjusted return expectations
4. Suggest diversification strategies
5. Recommend risk mitigation actions

Always structure your response with:
- Overall risk assessment (Low/Medium/High)
- Key risk metrics and their interpretation
- Risk factors affecting the portfolio
- Diversification analysis
- Specific risk management recommendations
- Stress test scenarios

Be thorough and provide quantitative risk assessments when possible. Focus on actionable risk management strategies."""
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate input for risk assessment."""
        if "request" not in state:
            return False
        
        request = state["request"]
        if not hasattr(request, 'risk') or not request.risk:
            return False
        
        risk_request = request.risk
        if not isinstance(risk_request, RiskRequest):
            return False
        
        if risk_request.portfolio_value <= 0:
            return False
        
        if risk_request.time_horizon <= 0 or risk_request.time_horizon > 50:
            return False
        
        return True
    
    def process_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process risk assessment request."""
        request = state["request"]
        risk_request = request.risk
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(risk_request, state)
        
        # Assess risk level
        risk_assessment = self._assess_risk_level(risk_request, risk_metrics)
        
        # Analyze diversification
        diversification = self._analyze_diversification(risk_request, state)
        
        # Generate risk recommendations
        recommendations = self._generate_risk_recommendations(risk_request, risk_metrics, risk_assessment)
        
        # Perform stress tests
        stress_tests = self._perform_stress_tests(risk_request, risk_metrics)
        
        return {
            "portfolio_value": risk_request.portfolio_value,
            "risk_tolerance": risk_request.risk_tolerance.value,
            "time_horizon": risk_request.time_horizon,
            "risk_metrics": risk_metrics,
            "overall_risk_assessment": risk_assessment,
            "diversification_analysis": diversification,
            "risk_recommendations": recommendations,
            "stress_test_results": stress_tests,
            "assessment_timestamp": datetime.now().isoformat(),
            "response": self._format_response(risk_request, risk_assessment, risk_metrics, recommendations)
        }
    
    def _calculate_risk_metrics(self, risk_request: RiskRequest, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        portfolio_value = risk_request.portfolio_value
        tolerance = risk_request.risk_tolerance
        horizon = risk_request.time_horizon
        
        # Simulate risk metrics (in real implementation, would use historical data)
        base_volatility = {
            RiskTolerance.CONSERVATIVE: 0.08,
            RiskTolerance.MODERATE: 0.15,
            RiskTolerance.AGGRESSIVE: 0.25
        }[tolerance]
        
        # Adjust for time horizon (longer horizon = more volatility)
        volatility = base_volatility * math.sqrt(horizon / 10)
        
        # Calculate Value at Risk (VaR) at 95% confidence
        var_95 = portfolio_value * volatility * 1.65
        
        # Calculate beta (market correlation)
        beta = 1.0 if tolerance == RiskTolerance.MODERATE else (
            0.7 if tolerance == RiskTolerance.CONSERVATIVE else 1.3
        )
        
        # Sharpe ratio (risk-adjusted return)
        expected_return = {
            RiskTolerance.CONSERVATIVE: 0.06,
            RiskTolerance.MODERATE: 0.08,
            RiskTolerance.AGGRESSIVE: 0.12
        }[tolerance]
        
        sharpe_ratio = (expected_return - 0.02) / volatility  # Assuming 2% risk-free rate
        
        # Maximum drawdown estimate
        max_drawdown = volatility * 2.5
        
        return {
            "volatility": volatility,
            "var_95": var_95,
            "beta": beta,
            "sharpe_ratio": sharpe_ratio,
            "expected_return": expected_return,
            "max_drawdown": max_drawdown,
            "risk_adjusted_return": expected_return / volatility
        }
    
    def _assess_risk_level(self, risk_request: RiskRequest, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk level."""
        tolerance = risk_request.risk_tolerance
        metrics = risk_metrics
        
        # Risk score (0-100)
        risk_score = min(100, metrics["volatility"] * 200)
        
        # Determine risk level relative to tolerance
        if tolerance == RiskTolerance.CONSERVATIVE:
            if risk_score < 30:
                level = "LOW"
                status = "APPROPRIATE"
            elif risk_score < 50:
                level = "MODERATE"
                status = "ACCEPTABLE"
            else:
                level = "HIGH"
                status = "EXCESSIVE"
        
        elif tolerance == RiskTolerance.MODERATE:
            if risk_score < 40:
                level = "LOW"
                status = "CONSERVATIVE"
            elif risk_score < 60:
                level = "MODERATE"
                status = "APPROPRIATE"
            else:
                level = "HIGH"
                status = "ELEVATED"
        
        else:  # AGGRESSIVE
            if risk_score < 50:
                level = "MODERATE"
                status = "CONSERVATIVE"
            elif risk_score < 70:
                level = "HIGH"
                status = "APPROPRIATE"
            else:
                level = "VERY HIGH"
                status = "AGGRESSIVE"
        
        return {
            "risk_level": level,
            "risk_score": risk_score,
            "status": status,
            "tolerance_match": status in ["APPROPRIATE", "ACCEPTABLE"],
            "key_factors": self._identify_risk_factors(risk_request, metrics)
        }
    
    def _analyze_diversification(self, risk_request: RiskRequest, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio diversification."""
        holdings = risk_request.current_holdings or []
        
        # Calculate concentration risk
        if not holdings:
            concentration_score = 0
            diversification_score = 50  # Neutral
        else:
            # Simple concentration calculation (in real implementation, would use weights)
            unique_holdings = len(set(holdings))
            concentration_score = max(0, 100 - (unique_holdings * 10))
            diversification_score = min(100, unique_holdings * 15)
        
        # Asset allocation analysis
        asset_classes = self._estimate_asset_classes(holdings)
        
        return {
            "concentration_risk": concentration_score,
            "diversification_score": diversification_score,
            "unique_holdings": len(set(holdings)) if holdings else 0,
            "asset_classes": asset_classes,
            "recommendations": self._get_diversification_recommendations(diversification_score)
        }
    
    def _generate_risk_recommendations(self, risk_request: RiskRequest, 
                                     risk_metrics: Dict[str, Any], 
                                     risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        tolerance = risk_request.risk_tolerance
        risk_level = risk_assessment["risk_level"]
        
        # Based on risk level vs tolerance
        if risk_assessment["status"] == "EXCESSIVE":
            recommendations.extend([
                "Consider reducing portfolio volatility through diversification",
                "Increase allocation to lower-risk assets (bonds, cash)",
                "Implement stop-loss orders to limit downside"
            ])
        elif risk_assessment["status"] == "CONSERVATIVE":
            recommendations.extend([
                "Consider increasing exposure to growth assets",
                "Add international diversification",
                "Explore sector-specific ETFs for targeted exposure"
            ])
        else:
            recommendations.extend([
                "Maintain current risk level with periodic reviews",
                "Rebalance portfolio quarterly to maintain target allocation",
                "Consider tax-loss harvesting opportunities"
            ])
        
        # General recommendations
        recommendations.extend([
            "Set up automatic portfolio rebalancing",
            "Monitor correlation between holdings",
            "Keep emergency fund separate from investment portfolio",
            "Review risk tolerance annually or after major life events"
        ])
        
        return recommendations
    
    def _perform_stress_tests(self, risk_request: RiskRequest, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress test scenarios."""
        portfolio_value = risk_request.portfolio_value
        volatility = risk_metrics["volatility"]
        beta = risk_metrics["beta"]
        
        scenarios = {
            "market_crash": {
                "description": "20% market decline",
                "impact": portfolio_value * 0.20 * beta,
                "probability": "Low (5% annual)"
            },
            "recession": {
                "description": "Economic recession",
                "impact": portfolio_value * 0.15 * beta,
                "probability": "Medium (20% annual)"
            },
            "inflation_spike": {
                "description": "High inflation environment",
                "impact": portfolio_value * 0.10,
                "probability": "Medium (15% annual)"
            },
            "sector_crash": {
                "description": "Technology sector decline 30%",
                "impact": portfolio_value * 0.30 * 0.3,  # Assuming 30% tech exposure
                "probability": "Low (10% annual)"
            }
        }
        
        # Calculate worst-case scenario
        worst_case = max(scenarios.values(), key=lambda x: x["impact"])
        
        return {
            "scenarios": scenarios,
            "worst_case_loss": worst_case["impact"],
            "worst_case_scenario": worst_case["description"],
            "portfolio_resilience": "STRONG" if worst_case["impact"] < portfolio_value * 0.25 else "MODERATE"
        }
    
    def _identify_risk_factors(self, risk_request: RiskRequest, risk_metrics: Dict[str, Any]) -> List[str]:
        """Identify key risk factors."""
        factors = []
        
        if risk_metrics["volatility"] > 0.2:
            factors.append("High portfolio volatility")
        
        if risk_metrics["beta"] > 1.2:
            factors.append("High market sensitivity")
        
        if risk_request.time_horizon < 5:
            factors.append("Short time horizon increases risk")
        
        if risk_request.portfolio_value > 1000000:
            factors.append("Large portfolio size requires active management")
        
        return factors
    
    def _estimate_asset_classes(self, holdings: List[str]) -> Dict[str, float]:
        """Estimate asset class allocation (simplified)."""
        if not holdings:
            return {"stocks": 0.6, "bonds": 0.3, "cash": 0.1}
        
        # Simple estimation based on common symbols
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        bond_symbols = ["TLT", "BND", "AGG"]
        
        stocks = sum(1 for h in holdings if any(s in h for s in stock_symbols))
        bonds = sum(1 for h in holdings if any(s in h for s in bond_symbols))
        
        total = len(holdings) or 1
        return {
            "stocks": stocks / total,
            "bonds": bonds / total,
            "cash": (total - stocks - bonds) / total
        }
    
    def _get_diversification_recommendations(self, score: int) -> List[str]:
        """Get diversification recommendations based on score."""
        if score < 30:
            return [
                "Urgent: Increase portfolio diversification",
                "Add at least 10 different holdings",
                "Consider index funds for instant diversification"
            ]
        elif score < 60:
            return [
                "Moderate diversification needed",
                "Add holdings from different sectors",
                "Consider international exposure"
            ]
        else:
            return [
                "Good diversification maintained",
                "Periodic rebalancing recommended",
                "Monitor concentration risk"
            ]
    
    def _format_response(self, risk_request: RiskRequest, risk_assessment: Dict[str, Any],
                        risk_metrics: Dict[str, Any], recommendations: List[str]) -> str:
        """Format the risk assessment response."""
        response_parts = [
            f"PORTFOLIO RISK ASSESSMENT",
            f"{'='*50}",
            f"Portfolio Value: ${risk_request.portfolio_value:,.2f}",
            f"Risk Tolerance: {risk_request.risk_tolerance.value.upper()}",
            f"Time Horizon: {risk_request.time_horizon} years",
            "",
            "OVERALL RISK ASSESSMENT:",
            f"Risk Level: {risk_assessment['risk_level']}",
            f"Risk Score: {risk_assessment['risk_score']}/100",
            f"Status: {risk_assessment['status']}",
            f"Tolerance Match: {'✓' if risk_assessment['tolerance_match'] else '✗'}",
            "",
            "KEY RISK METRICS:",
            f"Volatility: {risk_metrics['volatility']:.1%}",
            f"Value at Risk (95%): ${risk_metrics['var_95']:,.2f}",
            f"Beta: {risk_metrics['beta']:.2f}",
            f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}",
            f"Expected Annual Return: {risk_metrics['expected_return']:.1%}",
            f"Max Drawdown Estimate: {risk_metrics['max_drawdown']:.1%}",
            "",
            "RISK MANAGEMENT RECOMMENDATIONS:",
        ]
        
        for i, rec in enumerate(recommendations, 1):
            response_parts.append(f"{i}. {rec}")
        
        response_parts.extend([
            "",
            "STRESS TEST RESULTS:",
            f"Worst Case Loss: ${risk_assessment.get('stress_test_results', {}).get('worst_case_loss', 0):,.2f}",
            f"Portfolio Resilience: {risk_assessment.get('stress_test_results', {}).get('portfolio_resilience', 'UNKNOWN')}",
            "",
            f"Assessment completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return "\n".join(response_parts)
    
    def process_response(self, response: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM response with additional risk-specific analysis."""
        # Get base processing
        base_result = super().process_response(response, state)
        
        # Add risk-specific metrics
        request = state.get("request")
        if request and hasattr(request, 'risk'):
            risk_request = request.risk
            
            metrics.increment_counter('risk_assessments', 1, {'agent': self.config.name})
            metrics.increment_counter('risk_tolerance', 1, {
                'agent': self.config.name,
                'tolerance': risk_request.risk_tolerance.value
            })
            
            # Log portfolio value ranges
            if risk_request.portfolio_value < 100000:
                size_category = "small"
            elif risk_request.portfolio_value < 1000000:
                size_category = "medium"
            else:
                size_category = "large"
            
            metrics.increment_counter('portfolio_size', 1, {
                'agent': self.config.name,
                'size': size_category
            })
        
        return base_result


# Initialize the agent
risk_agent = RiskAgent()
