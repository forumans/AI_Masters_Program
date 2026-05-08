"""
Tax Agent - Tax Expert and Financial Advisor

PURPOSE:
--------
This agent provides tax advice and helps optimize tax situations.
It's like having a tax consultant available 24/7 to answer questions
about taxes, deductions, and tax-efficient strategies.

WHAT IT DOES:
-------------
- Answers tax questions based on current tax laws
- Identifies tax-saving opportunities
- Explains tax implications of financial decisions
- Suggests tax-efficient investment strategies
- Ensures compliance with tax regulations

HOW TO USE:
----------
Just ask a tax-related question and this agent will:
1. Analyze your tax situation
2. Identify potential deductions and credits
3. Provide actionable tax advice
4. Explain relevant tax laws
5. Suggest optimization strategies

IMPORTANT NOTE:
This is general tax advice. Always consult with a professional
tax advisor for your specific situation.
"""

from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseLangGraphAgent, AgentConfig, agent_registry
from ..models.request_models import TaxRequest
from ..utils.monitoring import get_logger, metrics

logger = get_logger(__name__)


class TaxAgent(BaseLangGraphAgent):
    """
    Tax Expert Agent
    
    This agent specializes in tax consultation and optimization,
    providing advice on tax laws, deductions, and tax-efficient strategies.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="tax_expert",
            description="Provides tax advice and optimization strategies",
            llm_model="gpt-3.5-turbo",
            max_tokens=1500,
            temperature=0.1,
            timeout=30.0,
            max_retries=3
        )
        super().__init__(config)
        agent_registry.register(self)
    
    def get_system_prompt(self) -> str:
        """Get system prompt for tax consultation."""
        return """You are a certified tax consultant with expertise in tax law, tax planning, and optimization strategies.

Your responsibilities:
1. Provide accurate tax advice based on current tax laws
2. Identify tax-saving opportunities
3. Explain tax implications of financial decisions
4. Suggest tax-efficient investment strategies
5. Ensure compliance with tax regulations

Always structure your response with:
- Direct answer to the tax question
- Tax-saving opportunities identified
- Relevant tax laws or regulations
- Actionable recommendations
- Risk considerations or limitations

Disclaimer: Include a note that this is general advice and consultation with a tax professional is recommended for specific situations.

Be thorough and provide practical, actionable tax advice."""
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate input for tax consultation."""
        if "request" not in state:
            return False
        
        request = state["request"]
        if not hasattr(request, 'tax') or not request.tax:
            return False
        
        tax_request = request.tax
        if not isinstance(tax_request, TaxRequest):
            return False
        
        if not tax_request.question or len(tax_request.question) < 10:
            return False
        
        return True
    
    def process_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process tax consultation request."""
        request = state["request"]
        tax_request = request.tax
        
        # Analyze the tax question
        analysis = self._analyze_tax_question(tax_request, state)
        
        # Identify tax-saving opportunities
        opportunities = self._identify_tax_opportunities(tax_request, state)
        
        # Generate recommendations
        recommendations = self._generate_tax_recommendations(tax_request, analysis, opportunities)
        
        return {
            "question_answered": tax_request.question,
            "tax_analysis": analysis,
            "tax_saving_opportunities": opportunities,
            "recommendations": recommendations,
            "jurisdiction": tax_request.jurisdiction,
            "tax_year": tax_request.year,
            "consultation_timestamp": datetime.now().isoformat(),
            "response": self._format_response(tax_request, analysis, opportunities, recommendations)
        }
    
    def _analyze_tax_question(self, tax_request: TaxRequest, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the tax question and categorize it."""
        question = tax_request.question.lower()
        
        # Categorize the tax question
        category = "general"
        if "capital gains" in question:
            category = "capital_gains"
        elif "dividend" in question:
            category = "dividends"
        elif "retirement" in question or "401k" in question or "ira" in question:
            category = "retirement"
        elif "loss" in question or "harvest" in question:
            category = "tax_loss_harvesting"
        elif "business" in question or "self-employed" in question:
            category = "business"
        
        # Determine complexity
        complexity = "simple"
        if any(word in question for word in ["complex", "multiple", "strategy", "optimization"]):
            complexity = "complex"
        
        return {
            "category": category,
            "complexity": complexity,
            "key_topics": self._extract_key_topics(question),
            "applicable_laws": self._get_applicable_tax_laws(category, tax_request.jurisdiction)
        }
    
    def _identify_tax_opportunities(self, tax_request: TaxRequest, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential tax-saving opportunities."""
        question = tax_request.question.lower()
        opportunities = []
        
        # Check for common tax-saving opportunities
        if "capital gains" in question:
            opportunities.append({
                "type": "tax_loss_harvesting",
                "potential_savings": "Up to 30% of gains",
                "description": "Sell losing investments to offset gains",
                "action_required": "Review portfolio for losing positions"
            })
        
        if "retirement" in question:
            opportunities.append({
                "type": "retirement_contributions",
                "potential_savings": "22-37% of contribution amount",
                "description": "Maximize tax-advantaged retirement accounts",
                "action_required": "Increase 401(k)/IRA contributions"
            })
        
        if any(word in question for word in ["investment", "portfolio", "stocks"]):
            opportunities.append({
                "type": "tax_efficient_funds",
                "potential_savings": "1-2% annually",
                "description": "Use tax-efficient investment vehicles",
                "action_required": "Consider ETFs or tax-managed funds"
            })
        
        # Always include general opportunities
        opportunities.extend([
            {
                "type": "itemized_deductions",
                "potential_savings": "Varies by income",
                "description": "Track and maximize itemized deductions",
                "action_required": "Maintain detailed records"
            },
            {
                "type": "tax_loss_carryforward",
                "potential_savings": "Future tax reduction",
                "description": "Carry forward excess losses to future years",
                "action_required": "Track capital loss carryforwards"
            }
        ])
        
        return opportunities
    
    def _generate_tax_recommendations(self, tax_request: TaxRequest, analysis: Dict[str, Any], 
                                    opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate specific tax recommendations."""
        recommendations = []
        
        # Based on analysis category
        if analysis["category"] == "capital_gains":
            recommendations.extend([
                "Consider tax-loss harvesting to offset capital gains",
                "Hold investments for over 1 year for long-term capital gains rates",
                "Review your tax bracket to optimize timing of sales"
            ])
        
        elif analysis["category"] == "retirement":
            recommendations.extend([
                "Maximize employer 401(k) match if available",
                "Consider Roth IRA conversion if in lower tax bracket",
                "Review required minimum distributions (RMDs) if applicable"
            ])
        
        # General recommendations
        recommendations.extend([
            "Consult with a tax professional for personalized advice",
            "Keep detailed records of all investment transactions",
            "Review tax withholding and adjust if necessary",
            "Consider quarterly estimated tax payments if needed"
        ])
        
        return recommendations
    
    def _extract_key_topics(self, question: str) -> List[str]:
        """Extract key topics from the tax question."""
        topics = []
        keywords = {
            "capital gains": ["capital gains", "gains", "profit"],
            "dividends": ["dividend", "distribution"],
            "retirement": ["retirement", "401k", "ira", "pension"],
            "deductions": ["deduction", "itemize", "standard"],
            "loss harvesting": ["loss", "harvest", "offset"],
            "business": ["business", "self-employed", "schedule c"]
        }
        
        for topic, words in keywords.items():
            if any(word in question for word in words):
                topics.append(topic)
        
        return topics
    
    def _get_applicable_tax_laws(self, category: str, jurisdiction: str) -> List[str]:
        """Get applicable tax laws based on category and jurisdiction."""
        if jurisdiction.upper() == "US":
            laws = {
                "capital_gains": [
                    "IRC Section 1221 - Capital Asset Definition",
                    "IRC Section 1222 - Capital Gains and Losses",
                    "IRC Section 1202 - Qualified Small Business Stock"
                ],
                "dividends": [
                    "IRC Section 1 - Tax Imposed",
                    "IRC Section 1(h) - Qualified Dividend Income"
                ],
                "retirement": [
                    "IRC Section 401(k) - Cash or Deferred Arrangements",
                    "IRC Section 408 - Individual Retirement Accounts"
                ]
            }
            return laws.get(category, ["Consult current tax code for specific provisions"])
        
        return ["Consult local tax regulations"]
    
    def _format_response(self, tax_request: TaxRequest, analysis: Dict[str, Any], 
                        opportunities: List[Dict[str, Any]], recommendations: List[str]) -> str:
        """Format the tax consultation response."""
        response_parts = [
            f"TAX CONSULTATION REPORT",
            f"{'='*50}",
            f"Jurisdiction: {tax_request.jurisdiction}",
            f"Tax Year: {tax_request.year}",
            f"Question: {tax_request.question}",
            "",
            "ANALYSIS:",
            f"Category: {analysis['category'].replace('_', ' ').title()}",
            f"Complexity: {analysis['complexity'].title()}",
            f"Key Topics: {', '.join(analysis['key_topics'])}",
            "",
            "TAX-SAVING OPPORTUNITIES:",
        ]
        
        for i, opp in enumerate(opportunities, 1):
            response_parts.extend([
                f"{i}. {opp['type'].replace('_', ' ').title()}",
                f"   Potential Savings: {opp['potential_savings']}",
                f"   Description: {opp['description']}",
                f"   Action Required: {opp['action_required']}",
                ""
            ])
        
        response_parts.extend([
            "RECOMMENDATIONS:",
        ])
        
        for i, rec in enumerate(recommendations, 1):
            response_parts.append(f"{i}. {rec}")
        
        response_parts.extend([
            "",
            "IMPORTANT DISCLAIMER:",
            "This tax advice is general in nature and may not apply to your specific situation. ",
            "Please consult with a qualified tax professional for personalized advice tailored to your circumstances.",
            "",
            f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return "\n".join(response_parts)
    
    def process_response(self, response: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM response with additional tax-specific analysis."""
        # Get base processing
        base_result = super().process_response(response, state)
        
        # Add tax-specific metrics
        request = state.get("request")
        if request and hasattr(request, 'tax'):
            tax_request = request.tax
            
            metrics.increment_counter('tax_questions_answered', 1, {'agent': self.config.name})
            metrics.increment_counter('tax_jurisdiction', 1, {
                'agent': self.config.name,
                'jurisdiction': tax_request.jurisdiction
            })
            
            # Count opportunities
            if 'tax_saving_opportunities' in base_result:
                metrics.histogram('tax_opportunities_identified', 
                                len(base_result['tax_saving_opportunities']),
                                {'agent': self.config.name})
        
        return base_result


# Initialize the agent
tax_agent = TaxAgent()
