"""
AutoGen-based Ticket Creator Agent for generating structured tickets
"""

import pandas as pd
import numpy as np
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from autogen_support import AssistantAgent, UserProxyAgent, autogen_is_ready, load_config_list

class TicketCreatorAgent:
    """
    AutoGen agent for creating structured tickets from analyzed feedback
    """
    
    def __init__(self, auto_approve: bool = False):
        """
        Initialize the Ticket Creator Agent
        
        Args:
            auto_approve (bool): Whether to auto-approve tickets
        """
        self.auto_approve = auto_approve
        self.logger = logging.getLogger(__name__)
        
        # Initialize AutoGen agents
        self._setup_autogen_agents()
        
        # Ticket types
        self.ticket_types = ['Bug', 'Feature Request', 'Improvement', 'Investigation', 'Documentation']
        
        # Ticket priorities
        self.ticket_priorities = ['Critical', 'High', 'Medium', 'Low']
        
        # Ticket statuses
        self.ticket_statuses = ['Open', 'In Progress', 'Pending', 'Resolved', 'Closed']
    
    def _setup_autogen_agents(self):
        """Setup AutoGen agents for ticket creation"""
        try:
            config_list = load_config_list(base_dir=os.path.join(os.path.dirname(__file__), '..', '..'))
            if not autogen_is_ready(config_list):
                raise RuntimeError("AutoGen dependencies or configuration are unavailable")
            
            # Create ticket creator assistant agent
            self.ticket_creator = AssistantAgent(
                name="ticket_creator",
                llm_config={
                    "config_list": config_list,
                    "temperature": 0.1,
                    "timeout": 120,
                },
                system_message="""You are an expert ticket creation specialist. 
                Your task is to create structured tickets from analyzed feedback.
                
                For each feedback item, create a ticket with:
                1. Ticket ID (unique identifier)
                2. Title (clear, concise summary)
                3. Description (detailed explanation)
                4. Type (Bug, Feature Request, etc.)
                5. Priority (Critical, High, Medium, Low)
                6. Status (initial status)
                7. Assignee (suggested team/role)
                8. Labels (relevant tags)
                9. Estimated effort (in story points or hours)
                10. Reproduction steps (for bugs)
                11. Expected outcome (for features)
                
                Provide a JSON response with all ticket fields.
                """
            )
            
            # Create user proxy agent
            self.user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            
            self.logger.info("Ticket creation AutoGen agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up ticket creation AutoGen agents: {str(e)}")
            # Fallback to rule-based creation
            self.ticket_creator = None
            self.user_proxy = None
    
    def create_ticket(self, feedback_data: Dict, analysis_data: Dict = None) -> Dict:
        """
        Create a structured ticket from feedback data
        
        Args:
            feedback_data (Dict): Original feedback data
            analysis_data (Dict): Analysis results (classification, bug analysis, etc.)
            
        Returns:
            Dict: Created ticket
        """
        try:
            # Try AutoGen creation first
            if self.ticket_creator and self.user_proxy:
                return self._create_with_autogen(feedback_data, analysis_data)
            else:
                return self._create_with_rules(feedback_data, analysis_data)
        except Exception as e:
            self.logger.error(f"Error creating ticket: {str(e)}")
            # Fallback to rule-based creation
            return self._create_with_rules(feedback_data, analysis_data)
    
    def _create_with_autogen(self, feedback_data: Dict, analysis_data: Dict = None) -> Dict:
        """
        Create ticket using AutoGen agents
        
        Args:
            feedback_data (Dict): Feedback data
            analysis_data (Dict): Analysis results
            
        Returns:
            Dict: Created ticket
        """
        try:
            # Prepare input for AutoGen
            input_text = f"""
            Feedback: {feedback_data.get('content', '')}
            Category: {analysis_data.get('predicted_category', '') if analysis_data else ''}
            Source: {feedback_data.get('source_type', '')}
            """
            
            if analysis_data:
                if 'bug_severity' in analysis_data:
                    input_text += f"\nBug Severity: {analysis_data['bug_severity']}"
                if 'feature_priority' in analysis_data:
                    input_text += f"\nFeature Priority: {analysis_data['feature_priority']}"
            
            message = f"Please create a structured ticket from this feedback: {input_text}"
            
            # Start the creation conversation
            chat_result = self.user_proxy.initiate_chat(
                self.ticket_creator,
                message=message,
                max_turns=2
            )
            
            # Extract the last response
            last_message = chat_result.chat_history[-1]['content']
            
            # Parse JSON response
            import json
            ticket_data = json.loads(last_message)
            
            # Validate and enhance ticket
            ticket_data = self._validate_and_enhance_ticket(ticket_data, feedback_data, analysis_data)
            
            return ticket_data
            
        except Exception as e:
            self.logger.error(f"AutoGen ticket creation failed: {str(e)}")
            # Fallback to rule-based
            return self._create_with_rules(feedback_data, analysis_data)
    
    def _create_with_rules(self, feedback_data: Dict, analysis_data: Dict = None) -> Dict:
        """
        Create ticket using rule-based approach
        
        Args:
            feedback_data (Dict): Feedback data
            analysis_data (Dict): Analysis results
            
        Returns:
            Dict: Created ticket
        """
        try:
            # Generate ticket ID
            ticket_id = self._generate_ticket_id()
            
            # Determine ticket type and priority
            ticket_type, priority = self._determine_ticket_type_and_priority(feedback_data, analysis_data)
            
            # Create title
            title = self._create_title(feedback_data, analysis_data)
            
            # Create description
            description = self._create_description(feedback_data, analysis_data)
            
            # Determine assignee
            assignee = self._determine_assignee(ticket_type, priority)
            
            # Generate labels
            labels = self._generate_labels(feedback_data, analysis_data, ticket_type)
            
            # Estimate effort
            effort = self._estimate_effort(ticket_type, priority, analysis_data)
            
            # Get reproduction steps or expected outcome
            reproduction_steps = self._get_reproduction_steps(analysis_data)
            expected_outcome = self._get_expected_outcome(analysis_data)
            
            # Create ticket
            ticket = {
                'ticket_id': ticket_id,
                'title': title,
                'description': description,
                'type': ticket_type,
                'priority': priority,
                'status': 'Open' if not self.auto_approve else 'In Progress',
                'assignee': assignee,
                'labels': labels,
                'estimated_effort': effort,
                'reproduction_steps': reproduction_steps,
                'expected_outcome': expected_outcome,
                'feedback_id': feedback_data.get('id', ''),
                'source_type': feedback_data.get('source_type', ''),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'confidence': analysis_data.get('confidence', 0.5) if analysis_data else 0.5
            }
            
            return ticket
            
        except Exception as e:
            self.logger.error(f"Rule-based ticket creation failed: {str(e)}")
            # Ultimate fallback
            return {
                'ticket_id': self._generate_ticket_id(),
                'title': 'Feedback Processing Required',
                'description': feedback_data.get('content', ''),
                'type': 'Investigation',
                'priority': 'Medium',
                'status': 'Open',
                'assignee': 'Team Lead',
                'labels': ['feedback'],
                'estimated_effort': '3',
                'reproduction_steps': '',
                'expected_outcome': '',
                'feedback_id': feedback_data.get('id', ''),
                'source_type': feedback_data.get('source_type', ''),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'confidence': 0.5
            }
    
    def _validate_and_enhance_ticket(self, ticket_data: Dict, feedback_data: Dict, analysis_data: Dict) -> Dict:
        """Validate and enhance ticket data"""
        # Ensure required fields
        if 'ticket_id' not in ticket_data:
            ticket_data['ticket_id'] = self._generate_ticket_id()
        
        if 'created_at' not in ticket_data:
            ticket_data['created_at'] = datetime.now().isoformat()
        
        # Add feedback linkage
        ticket_data['feedback_id'] = feedback_data.get('id', '')
        ticket_data['source_type'] = feedback_data.get('source_type', '')
        
        # Add confidence if not present
        if 'confidence' not in ticket_data:
            ticket_data['confidence'] = analysis_data.get('confidence', 0.5) if analysis_data else 0.5
        
        # Validate enum values
        if ticket_data.get('type') not in self.ticket_types:
            ticket_data['type'] = 'Investigation'
        
        if ticket_data.get('priority') not in self.ticket_priorities:
            ticket_data['priority'] = 'Medium'
        
        if ticket_data.get('status') not in self.ticket_statuses:
            ticket_data['status'] = 'Open'
        
        return ticket_data
    
    def _generate_ticket_id(self) -> str:
        """Generate unique ticket ID"""
        timestamp = datetime.now().strftime("%Y%m%d")
        random_suffix = str(uuid.uuid4())[:8].upper()
        return f"TK-{timestamp}-{random_suffix}"
    
    def _determine_ticket_type_and_priority(self, feedback_data: Dict, analysis_data: Dict) -> Tuple[str, str]:
        """Determine ticket type and priority"""
        # Default values
        ticket_type = 'Investigation'
        priority = 'Medium'
        
        # Check classification
        if analysis_data:
            category = analysis_data.get('predicted_category', '').lower()
            
            if 'bug' in category:
                ticket_type = 'Bug'
                # Get bug severity if available
                bug_severity = analysis_data.get('bug_severity', 'Medium')
                if bug_severity in ['Critical', 'High']:
                    priority = bug_severity
                else:
                    priority = 'Medium'
            
            elif 'feature' in category:
                ticket_type = 'Feature Request'
                # Get feature priority if available
                feature_priority = analysis_data.get('feature_priority', 'Medium')
                priority = feature_priority
            
            elif 'praise' in category:
                ticket_type = 'Documentation'  # Convert praise to documentation
                priority = 'Low'
            
            elif 'complaint' in category:
                ticket_type = 'Improvement'
                priority = 'High'
        
        return ticket_type, priority
    
    def _create_title(self, feedback_data: Dict, analysis_data: Dict) -> str:
        """Create ticket title"""
        content = feedback_data.get('content', '')
        
        # Extract first sentence or truncate
        if len(content) > 100:
            # Find first sentence
            first_period = content.find('.')
            if first_period > 20 and first_period < 100:
                title = content[:first_period + 1]
            else:
                title = content[:97] + '...'
        else:
            title = content
        
        # Add category prefix if available
        if analysis_data:
            category = analysis_data.get('predicted_category', '')
            if category:
                title = f"[{category}] {title}"
        
        return title.strip()
    
    def _create_description(self, feedback_data: Dict, analysis_data: Dict) -> str:
        """Create detailed ticket description"""
        description = f"**Feedback Details:**\n{feedback_data.get('content', '')}\n\n"
        description += f"**Source:** {feedback_data.get('source_type', 'Unknown')}\n"
        description += f"**Feedback ID:** {feedback_data.get('id', 'Unknown')}\n"
        
        if analysis_data:
            description += f"\n**Analysis Results:**\n"
            
            # Add classification info
            if 'predicted_category' in analysis_data:
                description += f"- Category: {analysis_data['predicted_category']}\n"
            
            # Add bug analysis info
            if 'bug_severity' in analysis_data:
                description += f"- Bug Severity: {analysis_data['bug_severity']}\n"
            if 'bug_category' in analysis_data:
                description += f"- Bug Category: {analysis_data['bug_category']}\n"
            if 'device_info' in analysis_data:
                description += f"- Device Info: {analysis_data['device_info']}\n"
            
            # Add feature analysis info
            if 'feature_priority' in analysis_data:
                description += f"- Feature Priority: {analysis_data['feature_priority']}\n"
            if 'target_users' in analysis_data:
                description += f"- Target Users: {analysis_data['target_users']}\n"
        
        return description
    
    def _determine_assignee(self, ticket_type: str, priority: str) -> str:
        """Determine suggested assignee"""
        assignee_map = {
            'Bug': {
                'Critical': 'Senior Developer',
                'High': 'Developer',
                'Medium': 'Developer',
                'Low': 'Junior Developer'
            },
            'Feature Request': {
                'Critical': 'Product Manager',
                'High': 'Senior Developer',
                'Medium': 'Developer',
                'Low': 'Junior Developer'
            },
            'Improvement': {
                'Critical': 'Product Manager',
                'High': 'UX Designer',
                'Medium': 'Developer',
                'Low': 'Junior Developer'
            },
            'Investigation': {
                'Critical': 'Team Lead',
                'High': 'Senior Developer',
                'Medium': 'Developer',
                'Low': 'Developer'
            },
            'Documentation': {
                'Critical': 'Technical Writer',
                'High': 'Technical Writer',
                'Medium': 'Junior Developer',
                'Low': 'Technical Writer'
            }
        }
        
        return assignee_map.get(ticket_type, {}).get(priority, 'Team Lead')
    
    def _generate_labels(self, feedback_data: Dict, analysis_data: Dict, ticket_type: str) -> List[str]:
        """Generate relevant labels for the ticket"""
        labels = [ticket_type.lower().replace(' ', '-')]
        
        # Add source label
        source_type = feedback_data.get('source_type', '').lower()
        if source_type:
            labels.append(source_type.replace('_', '-'))
        
        # Add analysis-based labels
        if analysis_data:
            # Bug-related labels
            if 'bug_category' in analysis_data:
                labels.append(analysis_data['bug_category'].lower())
            
            # Feature-related labels
            if 'feature_category' in analysis_data:
                labels.append(analysis_data['feature_category'].lower().replace('/', '-'))
        
        # Add priority label
        labels.append('user-feedback')
        
        return labels[:5]  # Limit to 5 labels
    
    def _estimate_effort(self, ticket_type: str, priority: str, analysis_data: Dict) -> str:
        """Estimate effort in story points"""
        effort_map = {
            'Bug': {
                'Critical': '8',
                'High': '5',
                'Medium': '3',
                'Low': '2'
            },
            'Feature Request': {
                'Critical': '13',
                'High': '8',
                'Medium': '5',
                'Low': '3'
            },
            'Improvement': {
                'Critical': '5',
                'High': '3',
                'Medium': '2',
                'Low': '1'
            },
            'Investigation': {
                'Critical': '5',
                'High': '3',
                'Medium': '2',
                'Low': '1'
            },
            'Documentation': {
                'Critical': '3',
                'High': '2',
                'Medium': '1',
                'Low': '1'
            }
        }
        
        base_effort = effort_map.get(ticket_type, {}).get(priority, '3')
        
        # Adjust based on complexity if available
        if analysis_data and 'implementation_complexity' in analysis_data:
            complexity = analysis_data['implementation_complexity']
            if complexity == 'High':
                base_effort = str(int(base_effort) * 2)
            elif complexity == 'Low':
                base_effort = str(max(1, int(base_effort) // 2))
        
        return base_effort
    
    def _get_reproduction_steps(self, analysis_data: Dict) -> str:
        """Get reproduction steps for bug tickets"""
        if analysis_data and 'reproduction_steps' in analysis_data:
            steps = analysis_data['reproduction_steps']
            if isinstance(steps, list):
                return '\n'.join(f"{i+1}. {step}" for i, step in enumerate(steps))
            elif isinstance(steps, str):
                return steps
        return ''
    
    def _get_expected_outcome(self, analysis_data: Dict) -> str:
        """Get expected outcome for feature tickets"""
        if analysis_data and 'expected_benefits' in analysis_data:
            benefits = analysis_data['expected_benefits']
            if isinstance(benefits, list):
                return '\n'.join(f"• {benefit}" for benefit in benefits)
            elif isinstance(benefits, str):
                return benefits.replace(';', '\n')
        return ''
    
    def create_batch_tickets(self, feedback_df: pd.DataFrame, analysis_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create tickets from a batch of feedback items
        
        Args:
            feedback_df (pd.DataFrame): Feedback data
            analysis_df (pd.DataFrame): Analysis results
            
        Returns:
            pd.DataFrame: Created tickets
        """
        try:
            tickets = []
            
            for idx, row in feedback_df.iterrows():
                feedback_data = row.to_dict()
                
                # Get corresponding analysis data
                analysis_data = None
                if analysis_df is not None:
                    analysis_row = analysis_df[analysis_df['id'] == row['id']]
                    if not analysis_row.empty:
                        analysis_data = analysis_row.iloc[0].to_dict()
                
                # Create ticket
                ticket = self.create_ticket(feedback_data, analysis_data)
                tickets.append(ticket)
                
                # Log progress
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"Created {idx + 1}/{len(feedback_df)} tickets")
            
            tickets_df = pd.DataFrame(tickets)
            self.logger.info(f"Successfully created {len(tickets_df)} tickets")
            return tickets_df
            
        except Exception as e:
            self.logger.error(f"Error in batch ticket creation: {str(e)}")
            raise
    
    def save_tickets(self, tickets_df: pd.DataFrame, output_path: str) -> bool:
        """
        Save tickets to CSV file
        
        Args:
            tickets_df (pd.DataFrame): Tickets data
            output_path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            tickets_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {len(tickets_df)} tickets to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving tickets: {str(e)}")
            return False
