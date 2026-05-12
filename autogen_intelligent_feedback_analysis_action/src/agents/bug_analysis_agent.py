"""
AutoGen-based Bug Analysis Agent for analyzing bug reports
"""

import pandas as pd
import numpy as np
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from autogen_support import AssistantAgent, UserProxyAgent, autogen_is_ready, load_config_list

class BugAnalysisAgent:
    """
    AutoGen agent for analyzing bug reports and extracting technical details
    """
    
    def __init__(self, severity_threshold: float = 0.8):
        """
        Initialize the Bug Analysis Agent
        
        Args:
            severity_threshold (float): Threshold for bug severity classification
        """
        self.severity_threshold = severity_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize AutoGen agents
        self._setup_autogen_agents()
        
        # Bug severity levels
        self.severity_levels = ['Critical', 'High', 'Medium', 'Low']
        
        # Common bug patterns
        self.bug_patterns = {
            'crash': ['crash', 'freez', 'hang', 'not responding', 'force close'],
            'performance': ['slow', 'lag', 'delay', 'performance', 'speed'],
            'ui': ['display', 'interface', 'layout', 'button', 'screen'],
            'functionality': ['not working', 'broken', 'feature not', 'unable to'],
            'compatibility': ['compatibility', 'version', 'device', 'os', 'browser']
        }
        
        # Device/OS patterns
        self.device_patterns = {
            'android': ['android', 'samsung', 'pixel', 'galaxy'],
            'ios': ['iphone', 'ipad', 'ios', 'apple'],
            'windows': ['windows', 'pc', 'desktop', 'laptop'],
            'mac': ['mac', 'macbook', 'imac', 'os x'],
            'web': ['browser', 'chrome', 'firefox', 'safari', 'edge']
        }
    
    def _setup_autogen_agents(self):
        """Setup AutoGen agents for bug analysis"""
        try:
            config_list = load_config_list(base_dir=os.path.join(os.path.dirname(__file__), '..', '..'))
            if not autogen_is_ready(config_list):
                raise RuntimeError("AutoGen dependencies or configuration are unavailable")
            
            # Create bug analyzer assistant agent
            self.bug_analyzer = AssistantAgent(
                name="bug_analyzer",
                llm_config={
                    "config_list": config_list,
                    "temperature": 0.1,
                    "timeout": 120,
                },
                system_message="""You are an expert bug analysis specialist. 
                Your task is to analyze bug reports and extract:
                1. Bug severity (Critical, High, Medium, Low)
                2. Bug category (crash, performance, UI, functionality, compatibility)
                3. Device/OS information
                4. Steps to reproduce (if available)
                5. Error messages (if any)
                
                Analyze the bug report and provide a JSON response with:
                {
                    "severity": "High",
                    "category": "crash",
                    "device_info": "Android 12 on Samsung Galaxy S21",
                    "reproduction_steps": ["Step 1", "Step 2"],
                    "error_message": "App crashes when syncing data",
                    "confidence": 0.85,
                    "reasoning": "User reports crash with specific device info"
                }
                """
            )
            
            # Create user proxy agent
            self.user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            
            self.logger.info("Bug analysis AutoGen agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up bug analysis AutoGen agents: {str(e)}")
            # Fallback to rule-based analysis
            self.bug_analyzer = None
            self.user_proxy = None
    
    def analyze_bug_report(self, feedback_text: str, feedback_id: str = "") -> Dict:
        """
        Analyze a single bug report
        
        Args:
            feedback_text (str): Bug report text to analyze
            feedback_id (str): ID of the feedback item
            
        Returns:
            Dict: Bug analysis result
        """
        try:
            # Try AutoGen analysis first
            if self.bug_analyzer and self.user_proxy:
                return self._analyze_with_autogen(feedback_text, feedback_id)
            else:
                return self._analyze_with_rules(feedback_text, feedback_id)
        except Exception as e:
            self.logger.error(f"Error analyzing bug report: {str(e)}")
            # Fallback to rule-based analysis
            return self._analyze_with_rules(feedback_text, feedback_id)
    
    def _analyze_with_autogen(self, feedback_text: str, feedback_id: str) -> Dict:
        """
        Analyze bug report using AutoGen agents
        
        Args:
            feedback_text (str): Bug report text
            feedback_id (str): Feedback ID
            
        Returns:
            Dict: Analysis result
        """
        try:
            message = f"Please analyze this bug report: {feedback_text}"
            
            # Start the analysis conversation
            chat_result = self.user_proxy.initiate_chat(
                self.bug_analyzer,
                message=message,
                max_turns=2
            )
            
            # Extract the last response
            last_message = chat_result.chat_history[-1]['content']
            
            # Parse JSON response
            import json
            result = json.loads(last_message)
            
            # Validate result
            if result['severity'] not in self.severity_levels:
                raise ValueError(f"Invalid severity: {result['severity']}")
            
            result['feedback_id'] = feedback_id
            return result
            
        except Exception as e:
            self.logger.error(f"AutoGen bug analysis failed: {str(e)}")
            # Fallback to rule-based
            return self._analyze_with_rules(feedback_text, feedback_id)
    
    def _analyze_with_rules(self, feedback_text: str, feedback_id: str) -> Dict:
        """
        Analyze bug report using rule-based approach
        
        Args:
            feedback_text (str): Bug report text
            feedback_id (str): Feedback ID
            
        Returns:
            Dict: Analysis result
        """
        try:
            text_lower = feedback_text.lower()
            
            # Determine bug category
            category_scores = {}
            for category, keywords in self.bug_patterns.items():
                score = 0
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1
                category_scores[category] = score
            
            best_category = max(category_scores, key=category_scores.get) if max(category_scores.values()) > 0 else 'functionality'
            
            # Determine severity
            severity = self._determine_severity(text_lower)
            
            # Extract device info
            device_info = self._extract_device_info(text_lower)
            
            # Extract error messages
            error_message = self._extract_error_message(feedback_text)
            
            # Extract reproduction steps
            reproduction_steps = self._extract_reproduction_steps(feedback_text)
            
            # Calculate confidence
            confidence = min((category_scores[best_category] + len(device_info.split())) / 5.0, 0.9)
            
            return {
                'feedback_id': feedback_id,
                'severity': severity,
                'category': best_category,
                'device_info': device_info,
                'reproduction_steps': reproduction_steps,
                'error_message': error_message,
                'confidence': confidence,
                'reasoning': f"Rule-based analysis: {best_category} category, {severity} severity"
            }
            
        except Exception as e:
            self.logger.error(f"Rule-based bug analysis failed: {str(e)}")
            # Ultimate fallback
            return {
                'feedback_id': feedback_id,
                'severity': 'Medium',
                'category': 'functionality',
                'device_info': 'Unknown',
                'reproduction_steps': [],
                'error_message': '',
                'confidence': 0.5,
                'reasoning': 'Default analysis due to error'
            }
    
    def _determine_severity(self, text_lower: str) -> str:
        """Determine bug severity based on text content"""
        critical_keywords = ['crash', 'critical', 'urgent', 'major', 'severe']
        high_keywords = ['broken', 'not working', 'unable', 'fail', 'error']
        medium_keywords = ['slow', 'lag', 'issue', 'problem', 'bug']
        
        if any(keyword in text_lower for keyword in critical_keywords):
            return 'Critical'
        elif any(keyword in text_lower for keyword in high_keywords):
            return 'High'
        elif any(keyword in text_lower for keyword in medium_keywords):
            return 'Medium'
        else:
            return 'Low'
    
    def _extract_device_info(self, text_lower: str) -> str:
        """Extract device/OS information from text"""
        for platform, keywords in self.device_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Look for version numbers
                    version_match = re.search(r'(\d+\.\d+)', text_lower)
                    version = f" {version_match.group(1)}" if version_match else ""
                    return f"{platform.title()}{version}"
        
        return "Unknown"
    
    def _extract_error_message(self, text: str) -> str:
        """Extract error messages from text"""
        # Look for quotes around error messages
        error_pattern = r'"([^"]*(?:error|exception|fail)[^"]*)"'
        match = re.search(error_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for common error indicators
        error_indicators = ['error:', 'exception:', 'failed:', 'cannot']
        for indicator in error_indicators:
            if indicator in text.lower():
                idx = text.lower().find(indicator)
                if idx != -1:
                    # Extract the rest of the sentence
                    end_idx = text.find('.', idx)
                    if end_idx != -1:
                        return text[idx:end_idx].strip()
        
        return ""
    
    def _extract_reproduction_steps(self, text: str) -> List[str]:
        """Extract reproduction steps from text"""
        steps = []
        
        # Look for numbered steps
        step_pattern = r'(\d+\.?\s*[^.!?]*[.!?])'
        matches = re.findall(step_pattern, text)
        
        for match in matches:
            step = match.strip()
            if len(step) > 5:  # Filter out very short steps
                steps.append(step)
        
        # If no numbered steps, look for step indicators
        if not steps:
            step_indicators = ['first', 'then', 'next', 'after', 'finally']
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(indicator in sentence.lower() for indicator in step_indicators):
                    if len(sentence) > 10:
                        steps.append(sentence + '.')
        
        return steps[:5]  # Limit to 5 steps
    
    def analyze_batch(self, bug_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze a batch of bug reports
        
        Args:
            bug_df (pd.DataFrame): DataFrame with bug reports
            
        Returns:
            pd.DataFrame: DataFrame with bug analysis results
        """
        try:
            results = []
            
            for idx, row in bug_df.iterrows():
                feedback_text = row['content']
                feedback_id = row['id']
                
                analysis = self.analyze_bug_report(feedback_text, feedback_id)
                
                result = {
                    'id': feedback_id,
                    'content': feedback_text,
                    'bug_severity': analysis['severity'],
                    'bug_category': analysis['category'],
                    'device_info': analysis['device_info'],
                    'reproduction_steps': ';'.join(analysis['reproduction_steps']),
                    'error_message': analysis['error_message'],
                    'analysis_confidence': analysis['confidence'],
                    'analysis_reasoning': analysis['reasoning'],
                    'source_type': row.get('source_type', 'unknown'),
                    'timestamp': row.get('timestamp', '')
                }
                
                results.append(result)
                
                # Log progress
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"Analyzed {idx + 1}/{len(bug_df)} bug reports")
            
            analyzed_df = pd.DataFrame(results)
            self.logger.info(f"Successfully analyzed {len(analyzed_df)} bug reports")
            return analyzed_df
            
        except Exception as e:
            self.logger.error(f"Error in batch bug analysis: {str(e)}")
            raise
    
    def get_severity_distribution(self, analyzed_df: pd.DataFrame) -> Dict:
        """
        Get distribution of bug severities
        
        Args:
            analyzed_df (pd.DataFrame): Analyzed bug data
            
        Returns:
            Dict: Severity distribution statistics
        """
        try:
            severity_counts = analyzed_df['bug_severity'].value_counts()
            total_bugs = len(analyzed_df)
            
            distribution = {}
            for severity in self.severity_levels:
                count = severity_counts.get(severity, 0)
                percentage = (count / total_bugs) * 100 if total_bugs > 0 else 0
                distribution[severity] = {
                    'count': count,
                    'percentage': percentage
                }
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error calculating severity distribution: {str(e)}")
            return {}
