"""
Equity Agent - Stock Market Expert

PURPOSE:
--------
This agent analyzes stocks and provides investment recommendations.
It's like having a professional stock analyst available 24/7.

WHAT IT DOES:
-------------
- Analyzes stocks using technical and fundamental analysis
- Provides BUY/HOLD/SELL recommendations
- Sets price targets and risk levels
- Gives confidence scores for recommendations

HOW TO USE:
----------
Just include stock symbols in your request and this agent will:
1. Analyze the stock's current situation
2. Look at market trends and technical indicators
3. Give a clear recommendation with reasoning
"""

from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseLangGraphAgent, AgentConfig, agent_registry
from ..models.request_models import StockRequest, AnalysisType
from ..utils.monitoring import get_logger, metrics

logger = get_logger(__name__)


class EquityAgent(BaseLangGraphAgent):
    """
    Stock Market Expert Agent
    
    This agent specializes in analyzing stocks and providing
    investment recommendations using both technical and fundamental analysis.
    """
    
    def __init__(self):
        """
        Set up the stock analysis agent with expert configuration.
        
        SETTINGS EXPLAINED:
        - name: "equity_expert" - For logging and monitoring
        - model: "gpt-3.5-turbo" - Fast and cost-effective for analysis
        - temperature: 0.1 - Low creativity for analytical work
        - max_tokens: 1500 - Enough for detailed analysis
        - timeout: 30s - Plenty of time for thorough analysis
        """
        config = AgentConfig(
            name="equity_expert",
            description="Analyzes stocks and provides investment recommendations",
            llm_model="gpt-3.5-turbo",
            max_tokens=1500,
            temperature=0.1,  # Low creativity for analytical work
            timeout=30.0,
            max_retries=3
        )
        super().__init__(config)
        agent_registry.register(self)  # Register so workflow can find this agent
    
    def get_system_prompt(self) -> str:
        """
        Define the AI's personality as a professional stock analyst.
        
        This tells the AI how to behave:
        - Act like an expert equity analyst
        - Use both technical and fundamental analysis
        - Give clear BUY/HOLD/SELL recommendations
        - Provide confidence levels
        - Structure responses consistently
        
        The prompt ensures consistent, professional analysis every time.
        """
        return """You are an expert equity analyst with deep knowledge of financial markets, technical analysis, and fundamental analysis.

Your responsibilities:
1. Analyze stocks using both technical and fundamental analysis
2. Provide clear investment recommendations (BUY, HOLD, SELL)
3. Assess risk levels and price targets
4. Consider market conditions and sector trends
5. Provide confidence scores for your recommendations

Always structure your response with:
- Overall recommendation (BUY/HOLD/SELL)
- Key analysis points
- Risk factors
- Price targets (if applicable)
- Confidence level (High/Moderate/Low)

Be thorough but concise. Focus on actionable insights for investors."""
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        Check if the request has stock symbols to analyze.
        
        WHAT IT CHECKS:
        - Must have 'stocks' in the request
        - Each stock must have a 'symbol'
        - Symbol must be a non-empty string
        
        Returns True if we can analyze the stocks, False otherwise.
        """
        # Check if we have stocks to analyze
        if "request" not in state:
            return False
        
        request = state["request"]
        if not hasattr(request, 'stocks') or not request.stocks:
            return False
        
        # Validate each stock
        for stock in request.stocks:
            if not isinstance(stock, StockRequest):
                return False
            if not stock.symbol or len(stock.symbol) < 1:
                return False
            if stock.analysis_type not in [AnalysisType.TECHNICAL, AnalysisType.FUNDAMENTAL, AnalysisType.BOTH]:
                return False
        
        return True
    
    def process_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze stocks and provide investment recommendations.
        
        WHAT IT DOES:
        1. Gets the stock symbols from the request
        2. Analyzes each stock using AI
        3. Calculates overall portfolio recommendation
        4. Returns structured results with recommendations
        
        LOGIC:
        - Analyzes each stock individually
        - Counts BUY vs SELL signals
        - Calculates average confidence
        - Gives overall recommendation if confidence is high
        """
        request = state["request"]
        stocks = request.stocks
        
        analyses = []
        overall_recommendation = "HOLD"  # Default to conservative
        total_confance = 0
        
        # Analyze each stock
        for stock in stocks:
            analysis = self._analyze_stock(stock, state)
            analyses.append(analysis)
            total_confance += analysis.get("confidence", 0.5)
        
        # Calculate overall portfolio recommendation
        if analyses:
            avg_confance = total_confance / len(analyses)
            buy_signals = sum(1 for a in analyses if a.get("recommendation") == "BUY")
            sell_signals = sum(1 for a in analyses if a.get("recommendation") == "SELL")
            
            # Only give strong recommendation if confidence is high
            if buy_signals > sell_signals and avg_confance > 0.7:
                overall_recommendation = "BUY"
            elif sell_signals > buy_signals and avg_confance > 0.7:
                overall_recommendation = "SELL"
        
        return {
            "stocks_analyzed": len(analyses),
            "stock_analyses": analyses,
            "portfolio_recommendation": overall_recommendation,
            "average_confidence": total_confance / len(analyses) if analyses else 0.5,
            "average_confidence": total_confidence / len(analyses) if analyses else 0.5,
            "analysis_timestamp": datetime.now().isoformat(),
            "response": self._format_response(analyses, overall_recommendation)
        }
    
    def _analyze_stock(self, stock: StockRequest, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single stock."""
        # In a real implementation, this would fetch real market data
        # For demo purposes, we'll simulate the analysis
        
        analysis_type = stock.analysis_type
        symbol = stock.symbol.upper()
        
        # Simulate analysis based on symbol (for demo)
        if symbol in ["AAPL", "MSFT", "GOOGL"]:
            recommendation = "BUY"
            confidence = 0.85
            price_target = f"${150 + hash(symbol) % 100}"
        elif symbol in ["TSLA", "AMZN", "META"]:
            recommendation = "HOLD"
            confidence = 0.70
            price_target = f"${100 + hash(symbol) % 80}"
        else:
            recommendation = "HOLD"
            confidence = 0.60
            price_target = f"${50 + hash(symbol) % 50}"
        
        return {
            "symbol": symbol,
            "analysis_type": analysis_type.value,
            "recommendation": recommendation,
            "confidence": confidence,
            "price_target": price_target,
            "key_points": [
                f"Strong {analysis_type} indicators",
                "Market conditions favorable",
                "Risk-reward ratio acceptable"
            ],
            "risk_factors": [
                "Market volatility",
                "Sector-specific risks",
                "Economic uncertainty"
            ]
        }
    
    def _format_response(self, analyses: List[Dict[str, Any]], overall_recommendation: str) -> str:
        """Format the analysis response."""
        response_parts = [
            f"EQUITY ANALYSIS REPORT",
            f"{'='*50}",
            f"Overall Portfolio Recommendation: {overall_recommendation}",
            f"Stocks Analyzed: {len(analyses)}",
            ""
        ]
        
        for analysis in analyses:
            response_parts.extend([
                f"Stock: {analysis['symbol']} ({analysis['analysis_type'].upper()})",
                f"Recommendation: {analysis['recommendation']} (Confidence: {analysis['confidence']:.0%})",
                f"Price Target: {analysis['price_target']}",
                "",
                "Key Points:",
                *[f"• {point}" for point in analysis['key_points']],
                "",
                "Risk Factors:",
                *[f"• {risk}" for risk in analysis['risk_factors']],
                "",
                "-"*40,
                ""
            ])
        
        response_parts.extend([
            "Summary:",
            f"- Portfolio recommendation: {overall_recommendation}",
            f"- Average confidence: {sum(a['confidence'] for a in analyses) / len(analyses):.0%}",
            "- Diversification recommended for risk management"
        ])
        
        return "\n".join(response_parts)
    
    def process_response(self, response: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM response with additional analysis."""
        # Get base processing
        base_result = super().process_response(response, state)
        
        # Add equity-specific metrics
        request = state.get("request")
        if request and hasattr(request, 'stocks'):
            metrics.increment_counter('stocks_analyzed', len(request.stocks), {'agent': self.config.name})
            
            for stock in request.stocks:
                metrics.increment_counter('analysis_type', 1, {
                    'agent': self.config.name,
                    'type': stock.analysis_type.value
                })
        
        return base_result


# Initialize the agent
equity_agent = EquityAgent()
