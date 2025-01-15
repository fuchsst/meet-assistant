import streamlit as st
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

def validate_project_key(key: str) -> bool:
    """Validate project key format."""
    return bool(re.match(r'^[A-Z][A-Z0-9-]+$', key))

def render_members_section():
    """Render and handle team members configuration."""
    if 'members' not in st.session_state:
        st.session_state.members = []
    
    # Form for adding new members
    with st.form(key="add_member_form"):
        st.write("Add team members:")
        cols = st.columns([3, 2, 3])
        with cols[0]:
            name = st.text_input("Name", key="new_member_name")
        with cols[1]:
            role = st.text_input("Role", key="new_member_role")
        with cols[2]:
            usernames = st.text_input("Usernames (comma-separated)", key="new_member_usernames")
        
        if st.form_submit_button("Add Member"):
            logger.info("Add member form submitted")
            if name and role:
                member_data = {
                    "name": name,
                    "role": role,
                    "user_names": [u.strip() for u in usernames.split(",")] if usernames else []
                }
                logger.info(f"Adding new member: {name} ({role})")
                st.session_state.members.append(member_data)
                st.experimental_rerun()
            else:
                logger.warning("Member creation failed: Missing required fields")
    
    # Display existing members
    for idx, member in enumerate(st.session_state.members):
        with st.form(key=f"member_form_{idx}"):
            cols = st.columns([3, 2, 3])
            with cols[0]:
                name = st.text_input("Name", value=member["name"], key=f"member_name_{idx}")
            with cols[1]:
                role = st.text_input("Role", value=member["role"], key=f"member_role_{idx}")
            with cols[2]:
                usernames = st.text_input(
                    "Usernames",
                    value=",".join(member.get("user_names", [])),
                    key=f"member_usernames_{idx}"
                )
            
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.form_submit_button("Remove"):
                    logger.info(f"Removing member: {member['name']} ({member['role']})")
                    st.session_state.members.pop(idx)
                    st.experimental_rerun()
            with col1:
                if st.form_submit_button("Update"):
                    logger.info(f"Updating member: {member['name']} -> {name}")
                    updated_member = {
                        "name": name,
                        "role": role,
                        "user_names": [u.strip() for u in usernames.split(",")] if usernames else []
                    }
                    st.session_state.members[idx] = updated_member
                    st.experimental_rerun()

def render_project_config_page():
    """Render the project configuration page."""
    st.header("Project Configuration")
    
    metadata_manager = st.session_state.metadata_manager
    
    # Determine if we're editing an existing project
    is_editing = hasattr(st.session_state, 'selected_project')
    current_project = st.session_state.selected_project if is_editing else None
    
    if metadata_manager.is_admin:
        # Get current config
        current_config = current_project.get('config', {}) if is_editing else {}
        
        # Initialize members in session state if editing
        if is_editing and 'members' not in st.session_state:
            st.session_state.members = current_config.get('members', [])
        
        # Project Information Form
        with st.form("project_form"):
            st.subheader("Project Information")
            
            # Project Key - only editable for new projects
            if is_editing:
                st.text_input(
                    "Project Key",
                    value=current_project['project_id'],
                    disabled=True,
                    help="Project key cannot be changed after creation"
                )
                project_key = current_project['project_id']
            else:
                project_key = st.text_input(
                    "Project Key",
                    help="Must start with a capital letter and contain only uppercase letters, numbers, and hyphens"
                )
                     
            # Project Name
            project_name = st.text_input(
                "Project Name",
                value=current_project['name'] if is_editing else "",
                key="project_name"
            )
            
            # Description
            project_description = st.text_area(
                "Description",
                value=current_project.get('description', '') if is_editing else "",
                key="project_description"
            )
            
            # Integration Settings - only show when editing
            if is_editing:
                st.subheader("Integration Settings")
                
                # Confluence Settings
                confluence_space = st.text_input(
                    "Space Key",
                    value=current_config.get('confluence', {}).get('space', ''),
                    key="confluence_space"
                )
                
                # Jira Settings
                jira_epics = st.text_area(
                    "Epic IDs (one per line)",
                    value="\n".join(e['id'] for e in current_config.get('jira', {}).get('epics', [])),
                    key="jira_epics"
                )
                
                # Slack Settings
                slack_channels = st.text_area(
                    "Channel IDs (one per line)",
                    value="\n".join(
                        f"{c['channel_id']},{c['name']}" 
                        for c in current_config.get('slack', {}).get('channels', [])
                    ),
                    help="Format: channel_id,channel_name",
                    key="slack_channels"
                )
                
                # Web Resources
                web_resources = st.text_area(
                    "Web Resources (one per line)",
                    value="\n".join(
                        f"{r['title']},{r['url']}" 
                        for r in current_config.get('web', [])
                    ),
                    help="Format: title,url",
                    key="web_resources"
                )
            
            # Submit button with logging before and after
            submit_label = "Update Project" if is_editing else "Create Project"
            logger.debug(f"Rendering form submit button: {submit_label}")
            submitted = st.form_submit_button(submit_label)
            logger.debug(f"Form submit button state: {submitted}")
            
            if submitted:
                logger.info("Project form submitted")
                logger.debug(f"Form data - Name: {project_name}, Key: {project_key}, Description: {project_description}")
                try:
                    # Validate required fields
                    if not project_name:
                        logger.warning("Project creation failed: Missing project name")
                        st.error("Project name is required")
                        return
                        
                    if not project_key:
                        logger.warning("Project creation failed: Missing project key")
                        st.error("Project key is required")
                        return
                        
                    if not validate_project_key(project_key):
                        logger.warning(f"Project creation failed: Invalid project key format: {project_key}")
                        st.error("Invalid project key format")
                        return
                        
                    logger.info(f"Creating project: {project_key} - {project_name}")
                    
                    # Prepare config data
                    if is_editing:
                        # For existing projects, include all integration settings
                        config = {
                            "members": st.session_state.members,
                            "confluence": {
                                "space": confluence_space,
                                "pages": []  # Pages will be fetched/updated separately
                            },
                            "jira": {
                                "epics": [
                                    {"id": epic_id.strip(), "title": ""}  # Titles will be fetched
                                    for epic_id in jira_epics.split("\n")
                                    if epic_id.strip()
                                ]
                            },
                            "slack": {
                                "channels": [
                                    {
                                        "channel_id": channel.split(",")[0].strip(),
                                        "name": channel.split(",")[1].strip() if "," in channel else "",
                                    }
                                    for channel in slack_channels.split("\n")
                                    if channel.strip()
                                ]
                            },
                            "web": [
                                {
                                    "title": resource.split(",")[0].strip(),
                                    "url": resource.split(",")[1].strip()
                                }
                                for resource in web_resources.split("\n")
                                if resource.strip() and "," in resource
                            ]
                        }
                    else:
                        # For new projects, just include basic config
                        config = {
                            "members": st.session_state.members if 'members' in st.session_state else [],
                            "confluence": {"space": "", "pages": []},
                            "jira": {"epics": []},
                            "slack": {"channels": []},
                            "web": []
                        }
                    
                    if is_editing:
                        # Update existing project
                        metadata_manager.update_project(
                            project_id=project_key,
                            updates={
                                'name': project_name,
                                'description': project_description,
                                'config': config
                            }
                        )
                        st.success(f"Project '{project_name}' updated successfully!")
                    else:
                        # Create new project
                        metadata_manager.create_project(
                            project_id=project_key,
                            name=project_name,
                            description=project_description,
                            config=config
                        )
                        st.success(f"Project '{project_name}' created successfully!")
                    
                    # Clear members from session state after successful save
                    if 'members' in st.session_state:
                        del st.session_state.members
                    
                    # Trigger page reload
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Failed to {'update' if is_editing else 'create'} project: {str(e)}")
        
        # Show team members section only when editing
        if is_editing:
            st.subheader("Team Members")
            render_members_section()
