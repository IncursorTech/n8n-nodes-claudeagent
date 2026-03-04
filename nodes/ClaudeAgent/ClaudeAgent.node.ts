import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';
import { NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';
import {
	query,
	tool,
	createSdkMcpServer,
	type SDKMessage,
	type SDKResultMessage,
	type SDKSystemMessage,
	type SDKAssistantMessage,
	type SDKUserMessage,
	type NonNullableUsage,
	type ModelUsage,
} from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

// ========== Output Format Type Definitions ==========
interface TextOutput {
	result: string;
}

interface SummaryOutput {
	session_id: string;
	success: boolean;
	result: string | null;
	error_type?: 'error_max_turns' | 'error_during_execution' | 'error_max_budget_usd';

	metrics: {
		turns: number;
		duration_ms: number;
		duration_api_ms: number;
		cost_usd: number;
	};

	usage: {
		input_tokens: number;
		output_tokens: number;
		cache_read_tokens: number;
		cache_creation_tokens: number;
	};

	model_usage: {
		[modelName: string]: {
			input_tokens: number;
			output_tokens: number;
			cache_read_tokens: number;
			cache_creation_tokens: number;
			cost_usd: number;
			web_search_requests: number;
		};
	};

	conversation: {
		user_messages: number;
		assistant_messages: number;
		tools_used: string[];
	};

	system: {
		model: string;
		cwd: string;
		permission_mode: string;
	};

	permission_denials?: Array<{
		tool_name: string;
		tool_use_id: string;
	}>;
}

interface StructuredOutput {
	structured_output: unknown;
	session_id: string;
	success: boolean;
	metrics: { turns: number; duration_ms: number; cost_usd: number };
}

interface FullOutput {
	messages: SDKMessage[];

	parsed: {
		session_id: string;

		init?: {
			model: string;
			permission_mode: string;
			cwd: string;
			tools: string[];
			mcp_servers: Array<{ name: string; status: string }>;
		};

		timeline: Array<{
			turn: number;
			user: string;
			assistant: string;
			tools: Array<{
				name: string;
				input: Record<string, unknown>;
				success: boolean;
				output?: string;
			}>;
		}>;

		result?: {
			success: boolean;
			text: string | null;
			error_type?: string;
			metrics: {
				turns: number;
				duration_ms: number;
				duration_api_ms: number;
				cost_usd: number;
			};
			usage: NonNullableUsage;
			model_usage: Record<string, ModelUsage>;
		};
	};
}

// ========== Helper Functions ==========
function extractUserText(msg: SDKUserMessage): string {
	const content = msg.message.content;
	if (typeof content === 'string') {
		return content;
	}
	if (Array.isArray(content)) {
		const textPart = content.find((c: any) => c.type === 'text');
		return textPart?.text || '';
	}
	return '';
}

function jsonSchemaToZodShape(schema: any): Record<string, z.ZodTypeAny> {
	const shape: Record<string, z.ZodTypeAny> = {};
	if (schema?.properties) {
		const required = new Set(schema.required || []);
		for (const [key, prop] of Object.entries<any>(schema.properties)) {
			let zodType: z.ZodTypeAny;
			switch (prop.type) {
				case 'number':
				case 'integer':
					zodType = z.number();
					break;
				case 'boolean':
					zodType = z.boolean();
					break;
				default:
					zodType = z.string();
			}
			if (prop.description) {
				zodType = zodType.describe(prop.description);
			}
			if (!required.has(key)) {
				zodType = zodType.optional();
			}
			shape[key] = zodType;
		}
	}
	return shape;
}

function formatAsText(messages: SDKMessage[]): TextOutput {
	const resultMsg = messages.find((m) => m.type === 'result') as SDKResultMessage | undefined;

	// Success
	if (resultMsg?.subtype === 'success' && resultMsg.result) {
		return { result: resultMsg.result };
	}

	// Error cases
	if (resultMsg?.subtype === 'error_max_turns') {
		return { result: 'Error: Maximum turns reached. Increase maxTurns or set to 0.' };
	}

	if (resultMsg?.subtype === 'error_during_execution') {
		return { result: 'Error: Execution failed. Enable debug mode for details.' };
	}

	if (resultMsg?.subtype === 'error_max_budget_usd') {
		return { result: 'Error: Maximum budget exceeded. Increase maxBudgetUsd or set to 0.' };
	}

	// Try to extract last assistant message
	const assistantMessages = messages.filter((m) => m.type === 'assistant') as SDKAssistantMessage[];
	if (assistantMessages.length > 0) {
		const last = assistantMessages[assistantMessages.length - 1];
		const textContent = last.message.content.find((c: any) => c.type === 'text');
		if (textContent?.text) {
			return { result: textContent.text };
		}
	}

	return { result: 'No response generated' };
}

function formatAsSummary(messages: SDKMessage[]): SummaryOutput {
	const resultMsg = messages.find((m) => m.type === 'result') as SDKResultMessage | undefined;
	const systemInit = messages.find((m) => m.type === 'system' && (m as any).subtype === 'init') as
		| SDKSystemMessage
		| undefined;

	// Count tool usage
	const toolsUsed = new Set<string>();
	messages.forEach((m) => {
		if (m.type === 'assistant') {
			const assistantMsg = m as SDKAssistantMessage;
			assistantMsg.message.content.forEach((content: any) => {
				if (content.type === 'tool_use') {
					toolsUsed.add(content.name);
				}
			});
		}
	});

	// Count messages
	const userMsgCount = messages.filter((m) => m.type === 'user').length;
	const assistantMsgCount = messages.filter((m) => m.type === 'assistant').length;

	return {
		session_id: resultMsg?.session_id || systemInit?.session_id || 'unknown',
		success: resultMsg?.subtype === 'success',
		result: resultMsg?.subtype === 'success' ? resultMsg.result : null,
		error_type: resultMsg?.subtype !== 'success' ? (resultMsg?.subtype as any) : undefined,

		metrics: {
			turns: resultMsg?.num_turns || 0,
			duration_ms: resultMsg?.duration_ms || 0,
			duration_api_ms: resultMsg?.duration_api_ms || 0,
			cost_usd: resultMsg?.total_cost_usd || 0,
		},

		usage: {
			input_tokens: resultMsg?.usage?.input_tokens || 0,
			output_tokens: resultMsg?.usage?.output_tokens || 0,
			cache_read_tokens: resultMsg?.usage?.cache_read_input_tokens || 0,
			cache_creation_tokens: resultMsg?.usage?.cache_creation_input_tokens || 0,
		},

		model_usage: resultMsg?.modelUsage
			? Object.entries(resultMsg.modelUsage).reduce(
					(acc, [modelName, usage]) => {
						acc[modelName] = {
							input_tokens: usage.inputTokens,
							output_tokens: usage.outputTokens,
							cache_read_tokens: usage.cacheReadInputTokens,
							cache_creation_tokens: usage.cacheCreationInputTokens,
							cost_usd: usage.costUSD,
							web_search_requests: usage.webSearchRequests,
						};
						return acc;
					},
					{} as SummaryOutput['model_usage'],
				)
			: {},

		conversation: {
			user_messages: userMsgCount,
			assistant_messages: assistantMsgCount,
			tools_used: Array.from(toolsUsed),
		},

		system: {
			model: systemInit?.model || 'unknown',
			cwd: systemInit?.cwd || '',
			permission_mode: systemInit?.permissionMode || 'unknown',
		},

		permission_denials:
			resultMsg?.permission_denials && resultMsg.permission_denials.length > 0
				? resultMsg.permission_denials.map((d) => ({
						tool_name: d.tool_name,
						tool_use_id: d.tool_use_id,
					}))
				: undefined,
	};
}

function formatAsStructured(messages: SDKMessage[]): StructuredOutput {
	const resultMsg = messages.find((m) => m.type === 'result') as SDKResultMessage | undefined;
	const systemInit = messages.find((m) => m.type === 'system' && (m as any).subtype === 'init') as
		| SDKSystemMessage
		| undefined;

	let structuredOutput: unknown = null;
	if (resultMsg?.subtype === 'success') {
		structuredOutput = resultMsg.structured_output ?? null;
	}

	return {
		structured_output: structuredOutput,
		session_id: resultMsg?.session_id || systemInit?.session_id || 'unknown',
		success: resultMsg?.subtype === 'success',
		metrics: {
			turns: resultMsg?.num_turns || 0,
			duration_ms: resultMsg?.duration_ms || 0,
			cost_usd: resultMsg?.total_cost_usd || 0,
		},
	};
}

function formatAsFull(messages: SDKMessage[]): FullOutput {
	const systemInit = messages.find((m) => m.type === 'system' && (m as any).subtype === 'init') as
		| SDKSystemMessage
		| undefined;

	const resultMsg = messages.find((m) => m.type === 'result') as SDKResultMessage | undefined;

	// Build timeline
	const timeline: FullOutput['parsed']['timeline'] = [];
	let currentTurn: any = null;

	messages.forEach((m) => {
		if (m.type === 'user' && !(m as any).isSynthetic) {
			// New user message = new turn
			currentTurn = {
				turn: timeline.length + 1,
				user: extractUserText(m as SDKUserMessage),
				assistant: '',
				tools: [],
			};
			timeline.push(currentTurn);
		} else if (m.type === 'assistant' && currentTurn) {
			const assistantMsg = m as SDKAssistantMessage;

			// Extract text
			const textContent = assistantMsg.message.content.find((c: any) => c.type === 'text');
			if (textContent) {
				currentTurn.assistant = (textContent as any).text;
			}

			// Extract tool usage
			assistantMsg.message.content.forEach((content: any) => {
				if (content.type === 'tool_use') {
					currentTurn.tools.push({
						name: content.name,
						input: content.input,
						success: true,
						output: undefined,
					});
				}
			});
		}
	});

	return {
		messages,

		parsed: {
			session_id: resultMsg?.session_id || systemInit?.session_id || 'unknown',

			init: systemInit
				? {
						model: systemInit.model,
						permission_mode: systemInit.permissionMode,
						cwd: systemInit.cwd,
						tools: systemInit.tools,
						mcp_servers: systemInit.mcp_servers,
					}
				: undefined,

			timeline,

			result: resultMsg
				? {
						success: resultMsg.subtype === 'success',
						text: resultMsg.subtype === 'success' ? resultMsg.result : null,
						error_type: resultMsg.subtype !== 'success' ? resultMsg.subtype : undefined,
						metrics: {
							turns: resultMsg.num_turns,
							duration_ms: resultMsg.duration_ms,
							duration_api_ms: resultMsg.duration_api_ms,
							cost_usd: resultMsg.total_cost_usd,
						},
						usage: resultMsg.usage,
						model_usage: resultMsg.modelUsage,
					}
				: undefined,
		},
	};
}

// ========== Node Class Definition ==========
export class ClaudeAgent implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Claude Agent',
		name: 'claudeAgent',
		icon: 'file:claudeagent.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}: {{$parameter["prompt"]}}',
		description: 'Execute AI-powered coding tasks with Claude Agent SDK',
		defaults: {
			name: 'Claude Agent',
		},
		inputs: [
			{ displayName: 'Data', type: NodeConnectionTypes.Main },
			{ displayName: '', type: NodeConnectionTypes.AiTool },
		],
		outputs: [{ type: NodeConnectionTypes.Main }],
		properties: [
			// ============ Session Management ============
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'New Query',
						value: 'query',
						description: 'Start a new conversation',
						action: 'Start a new conversation',
					},
					{
						name: 'Continue',
						value: 'continue',
						description: 'Continue the most recent session',
						action: 'Continue most recent session',
					},
					{
						name: 'Resume',
						value: 'resume',
						description: 'Resume a specific session by ID',
						action: 'Resume specific session',
					},
					{
						name: 'Fork',
						value: 'fork',
						description: 'Fork from a session (new session ID)',
						action: 'Fork from session',
					},
				],
				default: 'query',
				description: 'How to handle the conversation session',
			},

			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				default: '',
				required: true,
				displayOptions: {
					show: {
						operation: ['resume', 'fork'],
					},
				},
				description: 'The session ID to resume or fork from',
				placeholder: '550e8400-e29b-41d4-a716-446655440000',
			},

			// ============ Prompt ============
			{
				displayName: 'Prompt',
				name: 'prompt',
				type: 'string',
				typeOptions: {
					rows: 5,
				},
				default: '',
				required: true,
				description: 'The instruction to send to Claude Agent',
				placeholder: 'Create a Python function to parse CSV files and extract email addresses',
			},

			// ============ Working Environment ============
			{
				displayName: 'Working Directory',
				name: 'cwd',
				type: 'string',
				default: '',
				description: 'Directory where Claude Agent operates',
				placeholder: '/path/to/project',
				hint: 'Leave empty to use current directory',
			},

			// ============ Model Configuration ============
			{
				displayName: 'Model',
				name: 'model',
				type: 'options',
				options: [
					{
						name: 'Sonnet',
						value: 'sonnet',
						description: 'Fast and efficient (recommended)',
					},
					{
						name: 'Opus',
						value: 'opus',
						description: 'Most capable for complex tasks',
					},
					{
						name: 'Haiku',
						value: 'haiku',
						description: 'Fastest and most cost-effective',
					},
				],
				default: 'sonnet',
				description: 'Claude model to use',
			},

			{
				displayName: 'Permission Mode',
				name: 'permissionMode',
				type: 'options',
				options: [
					{
						name: 'Bypass All',
						value: 'bypassPermissions',
						description: 'No prompts, full automation (recommended for n8n)',
					},
					{
						name: 'Accept Edits',
						value: 'acceptEdits',
						description: 'Auto-accept file edits only',
					},
					{
						name: 'Ask Always',
						value: 'default',
						description: 'Prompt for all operations',
					},
					{
						name: 'Plan Mode',
						value: 'plan',
						description: 'Plan first, then execute',
					},
					{
						name: "Don't Ask",
						value: 'dontAsk',
						description: "Deny operations that aren't pre-approved",
					},
					{
						name: 'Delegate',
						value: 'delegate',
						description: 'Restrict to Teammate and Task tools only',
					},
				],
				default: 'bypassPermissions',
				description: 'How to handle tool permissions',
			},

			// ============ Execution Limits ============
			{
				displayName: 'Max Turns',
				name: 'maxTurns',
				type: 'number',
				default: 0,
				description: 'Maximum conversation turns',
				hint: '0 = unlimited (recommended). Complex tasks may need 50+ turns.',
			},

			{
				displayName: 'Timeout (Seconds)',
				name: 'timeout',
				type: 'number',
				default: 0,
				description: 'Maximum execution time',
				hint: '0 = unlimited (recommended)',
			},

			{
				displayName: 'Max Budget (USD)',
				name: 'maxBudgetUsd',
				type: 'number',
				default: 0,
				hint: '0 = unlimited',
				typeOptions: { minValue: 0, numberPrecision: 2 },
			},

			// ============ Output Configuration ============
			{
				displayName: 'Output Format',
				name: 'outputFormat',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Summary',
						value: 'summary',
						description: 'Key metrics and result (recommended)',
					},
					{
						name: 'Full',
						value: 'full',
						description: 'Complete conversation with all messages',
					},
					{
						name: 'Text Only',
						value: 'text',
						description: 'Just the final result text',
					},
					{
						name: 'Structured (JSON Schema)',
						value: 'structured',
						description: 'Structured JSON output matching a schema',
					},
				],
				default: 'summary',
				description: 'How to format the output',
			},

			{
				displayName: 'Output JSON Schema',
				name: 'outputJsonSchema',
				type: 'json',
				default:
					'{\n  "type": "object",\n  "properties": {\n    "result": { "type": "string" }\n  },\n  "required": ["result"]\n}',
				required: true,
				displayOptions: {
					show: {
						outputFormat: ['structured'],
					},
				},
			},

			// ============ Advanced Options ============
			{
				displayName: 'Advanced Options',
				name: 'advancedOptions',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				options: [
					// --- System Prompt ---
					{
						displayName: 'System Prompt Mode',
						name: 'systemPromptMode',
						type: 'options',
						options: [
							{
								name: 'Default',
								value: 'default',
								description: 'Use Claude Agent preset',
							},
							{
								name: 'Append',
								value: 'append',
								description: 'Add to default preset',
							},
							{
								name: 'Custom',
								value: 'custom',
								description: 'Replace completely',
							},
						],
						default: 'default',
					},
					{
						displayName: 'Append System Prompt',
						name: 'appendSystemPrompt',
						type: 'string',
						typeOptions: { rows: 3 },
						default: '',
						placeholder: 'Focus on clean, well-documented code...',
						description: 'Additional instructions appended to default',
					},
					{
						displayName: 'Custom System Prompt',
						name: 'customSystemPrompt',
						type: 'string',
						typeOptions: { rows: 4 },
						default: '',
						placeholder: 'You are a Python expert...',
						description: 'Completely replace the default system prompt',
					},

					// --- Model Configuration ---
					{
						displayName: 'Fallback Model',
						name: 'fallbackModel',
						type: 'options',
						options: [
							{ name: 'None', value: '' },
							{ name: 'Sonnet', value: 'sonnet' },
							{ name: 'Opus', value: 'opus' },
							{ name: 'Haiku', value: 'haiku' },
						],
						default: '',
						description: 'Switch to this if primary model is overloaded',
					},
					{
						displayName: 'Max Thinking Tokens',
						name: 'maxThinkingTokens',
						type: 'number',
						default: 0,
						description: 'Limit extended thinking (0 = unlimited)',
					},

					// --- Tool Permissions ---
					{
						displayName: 'Allowed Tools',
						name: 'allowedTools',
						type: 'multiOptions',
						options: [
							{ name: 'AskUserQuestion', value: 'AskUserQuestion' },
							{ name: 'Bash', value: 'Bash' },
							{ name: 'Config', value: 'Config' },
							{ name: 'Edit', value: 'Edit' },
							{ name: 'Glob', value: 'Glob' },
							{ name: 'Grep', value: 'Grep' },
							{ name: 'KillShell', value: 'KillShell' },
							{ name: 'ListMcpResources', value: 'ListMcpResources' },
							{ name: 'NotebookEdit', value: 'NotebookEdit' },
							{ name: 'Read', value: 'Read' },
							{ name: 'ReadMcpResource', value: 'ReadMcpResource' },
							{ name: 'Task', value: 'Task' },
							{ name: 'TaskOutput', value: 'TaskOutput' },
							{ name: 'TodoWrite', value: 'TodoWrite' },
							{ name: 'WebFetch', value: 'WebFetch' },
							{ name: 'WebSearch', value: 'WebSearch' },
							{ name: 'Write', value: 'Write' },
						],
						default: [],
						description: 'Limit to specific tools (empty = allow all)',
					},
					{
						displayName: 'Disallowed Tools',
						name: 'disallowedTools',
						type: 'multiOptions',
						options: [
							{ name: 'AskUserQuestion', value: 'AskUserQuestion' },
							{ name: 'Bash', value: 'Bash' },
							{ name: 'Config', value: 'Config' },
							{ name: 'Edit', value: 'Edit' },
							{ name: 'Glob', value: 'Glob' },
							{ name: 'Grep', value: 'Grep' },
							{ name: 'KillShell', value: 'KillShell' },
							{ name: 'ListMcpResources', value: 'ListMcpResources' },
							{ name: 'NotebookEdit', value: 'NotebookEdit' },
							{ name: 'Read', value: 'Read' },
							{ name: 'ReadMcpResource', value: 'ReadMcpResource' },
							{ name: 'Task', value: 'Task' },
							{ name: 'TaskOutput', value: 'TaskOutput' },
							{ name: 'TodoWrite', value: 'TodoWrite' },
							{ name: 'WebFetch', value: 'WebFetch' },
							{ name: 'WebSearch', value: 'WebSearch' },
							{ name: 'Write', value: 'Write' },
						],
						default: [],
						description: 'Block specific tools (overrides allowed)',
					},
					{
						displayName: 'Additional Directories',
						name: 'additionalDirectories',
						type: 'fixedCollection',
						typeOptions: {
							multipleValues: true,
						},
						default: {},
						placeholder: 'Add Directory',
						description: 'Extra directories to grant access',
						options: [
							{
								displayName: 'Directories',
								name: 'directories',
								values: [
									{
										displayName: 'Directory Path',
										name: 'path',
										type: 'string',
										default: '',
										placeholder: '/path/to/directory',
										description: 'Absolute path to the directory',
									},
								],
							},
						],
					},

					// --- Environment Variables ---
					{
						displayName: 'Environment Variables',
						name: 'environmentVariables',
						type: 'fixedCollection',
						typeOptions: { multipleValues: true },
						default: {},
						placeholder: 'Add Variable',
						options: [
							{
								displayName: 'Variables',
								name: 'variables',
								values: [
									{
										displayName: 'Name',
										name: 'name',
										type: 'string',
										default: '',
									},
									{
										displayName: 'Value',
										name: 'value',
										type: 'string',
										default: '',
									},
								],
							},
						],
					},

					// --- Session ---
					{
						displayName: 'Persist Session',
						name: 'persistSession',
						type: 'boolean',
						default: true,
					},

					// --- Settings Sources ---
					{
						displayName: 'Settings Sources',
						name: 'settingSources',
						type: 'multiOptions',
						options: [
							{
								name: 'User (~/.claude/settings.json)',
								value: 'user',
							},
							{
								name: 'Project (.claude/settings.json + CLAUDE.md)',
								value: 'project',
							},
							{
								name: 'Local (.claude/settings.local.json)',
								value: 'local',
							},
						],
						default: [],
					},

					// --- Debug ---
					{
						displayName: 'Debug Mode',
						name: 'debug',
						type: 'boolean',
						default: false,
						description: 'Enable detailed logging',
					},
					{
						displayName: 'Include Stream Events',
						name: 'includePartialMessages',
						type: 'boolean',
						default: false,
						description: 'Include real-time streaming events (full format only)',
					},
				],
			},

			// ============ MCP Servers ============
			{
				displayName: 'MCP Servers',
				name: 'mcpServers',
				type: 'fixedCollection',
				typeOptions: { multipleValues: true },
				default: {},
				placeholder: 'Add MCP Server',
				options: [
					{
						displayName: 'Servers',
						name: 'servers',
						values: [
							{
								displayName: 'Server Name',
								name: 'name',
								type: 'string',
								default: '',
								placeholder: 'my-server',
							},
							{
								displayName: 'Transport Type',
								name: 'transportType',
								type: 'options',
								options: [
									{ name: 'stdio', value: 'stdio' },
									{ name: 'SSE', value: 'sse' },
									{ name: 'HTTP', value: 'http' },
								],
								default: 'stdio',
							},
							{
								displayName: 'Command',
								name: 'command',
								type: 'string',
								default: '',
								placeholder: 'npx',
								displayOptions: {
									show: {
										transportType: ['stdio'],
									},
								},
							},
							{
								displayName: 'Arguments',
								name: 'args',
								type: 'string',
								default: '',
								placeholder: '@modelcontextprotocol/server-filesystem /tmp',
								description: 'Space-separated arguments',
								displayOptions: {
									show: {
										transportType: ['stdio'],
									},
								},
							},
							{
								displayName: 'Environment (JSON)',
								name: 'env',
								type: 'json',
								default: '{}',
								displayOptions: {
									show: {
										transportType: ['stdio'],
									},
								},
							},
							{
								displayName: 'URL',
								name: 'url',
								type: 'string',
								default: '',
								placeholder: 'https://mcp.example.com/sse',
								displayOptions: {
									show: {
										transportType: ['sse', 'http'],
									},
								},
							},
							{
								displayName: 'Headers (JSON)',
								name: 'headers',
								type: 'json',
								default: '{}',
								displayOptions: {
									show: {
										transportType: ['sse', 'http'],
									},
								},
							},
						],
					},
				],
			},

			// ============ Custom Tools ============
			{
				displayName: 'Custom Tools',
				name: 'customTools',
				type: 'fixedCollection',
				typeOptions: { multipleValues: true },
				default: {},
				placeholder: 'Add Custom Tool',
				options: [
					{
						displayName: 'Tools',
						name: 'tools',
						values: [
							{
								displayName: 'Tool Name',
								name: 'name',
								type: 'string',
								default: '',
								placeholder: 'get_weather',
							},
							{
								displayName: 'Description',
								name: 'description',
								type: 'string',
								typeOptions: { rows: 2 },
								default: '',
								placeholder: 'Get current weather for a city',
							},
							{
								displayName: 'Input Schema (JSON)',
								name: 'inputSchema',
								type: 'json',
								default:
									'{\n  "type": "object",\n  "properties": {\n    "query": { "type": "string", "description": "Input query" }\n  },\n  "required": ["query"]\n}',
							},
							{
								displayName: 'Static Response',
								name: 'staticResponse',
								type: 'string',
								typeOptions: { rows: 3 },
								default: '',
								placeholder: 'Tool response content...',
							},
						],
					},
				],
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
			try {
				// ========== Get Parameters ==========
				const operation = this.getNodeParameter('operation', itemIndex) as string;
				const prompt = this.getNodeParameter('prompt', itemIndex) as string;
				const cwd = this.getNodeParameter('cwd', itemIndex, '') as string;
				const model = this.getNodeParameter('model', itemIndex) as string;
				const permissionMode = this.getNodeParameter('permissionMode', itemIndex) as string;
				const maxTurns = this.getNodeParameter('maxTurns', itemIndex) as number;
				const timeout = this.getNodeParameter('timeout', itemIndex) as number;
				const maxBudgetUsd = this.getNodeParameter('maxBudgetUsd', itemIndex, 0) as number;
				const outputFormat = this.getNodeParameter('outputFormat', itemIndex) as string;

				const advancedOptions = this.getNodeParameter('advancedOptions', itemIndex, {}) as {
					systemPromptMode?: string;
					appendSystemPrompt?: string;
					customSystemPrompt?: string;
					fallbackModel?: string;
					maxThinkingTokens?: number;
					allowedTools?: string[];
					disallowedTools?: string[];
					additionalDirectories?: {
						directories?: Array<{ path: string }>;
					};
					environmentVariables?: {
						variables?: Array<{ name: string; value: string }>;
					};
					persistSession?: boolean;
					settingSources?: string[];
					debug?: boolean;
					includePartialMessages?: boolean;
				};

				// ========== Validation ==========
				if (!prompt?.trim()) {
					throw new NodeOperationError(this.getNode(), 'Prompt is required', { itemIndex });
				}

				// ========== Build Query Options ==========
				const queryOptions: {
					prompt: string;
					options: any;
				} = {
					prompt,
					options: {
						model,
						permissionMode: permissionMode as any,
					},
				};

				// Bypass permissions safety flag
				if (permissionMode === 'bypassPermissions') {
					queryOptions.options.allowDangerouslySkipPermissions = true;
				}

				// Working directory
				if (cwd?.trim()) {
					queryOptions.options.cwd = cwd.trim();
				}

				// Session management
				switch (operation) {
					case 'continue':
						queryOptions.options.continue = true;
						break;

					case 'resume': {
						const resumeSessionId = this.getNodeParameter('sessionId', itemIndex) as string;
						if (!resumeSessionId?.trim()) {
							throw new NodeOperationError(
								this.getNode(),
								'Session ID is required for resume operation',
								{ itemIndex },
							);
						}
						queryOptions.options.resume = resumeSessionId.trim();
						break;
					}

					case 'fork': {
						const forkSessionId = this.getNodeParameter('sessionId', itemIndex) as string;
						if (!forkSessionId?.trim()) {
							throw new NodeOperationError(
								this.getNode(),
								'Session ID is required for fork operation',
								{ itemIndex },
							);
						}
						queryOptions.options.resume = forkSessionId.trim();
						queryOptions.options.forkSession = true;
						break;
					}
				}

				// Execution limits
				if (maxTurns > 0) {
					queryOptions.options.maxTurns = maxTurns;
				}

				if (timeout > 0) {
					const abortController = new AbortController();
					setTimeout(() => abortController.abort(), timeout * 1000);
					queryOptions.options.abortController = abortController;
				}

				if (maxBudgetUsd > 0) {
					queryOptions.options.maxBudgetUsd = maxBudgetUsd;
				}

				// Structured output
				if (outputFormat === 'structured') {
					const outputJsonSchema = this.getNodeParameter(
						'outputJsonSchema',
						itemIndex,
					) as string;
					try {
						const schema = JSON.parse(outputJsonSchema);
						queryOptions.options.outputFormat = { type: 'json_schema', schema };
					} catch {
						throw new NodeOperationError(
							this.getNode(),
							'Invalid JSON schema for structured output',
							{ itemIndex },
						);
					}
				}

				// System prompt
				const systemPromptMode = advancedOptions.systemPromptMode || 'default';
				switch (systemPromptMode) {
					case 'append':
						if (advancedOptions.appendSystemPrompt?.trim()) {
							queryOptions.options.systemPrompt = {
								type: 'preset',
								preset: 'claude_code',
								append: advancedOptions.appendSystemPrompt.trim(),
							};
						}
						break;
					case 'custom':
						if (advancedOptions.customSystemPrompt?.trim()) {
							queryOptions.options.systemPrompt =
								advancedOptions.customSystemPrompt.trim();
						}
						break;
				}

				// Model configuration
				if (advancedOptions.fallbackModel) {
					queryOptions.options.fallbackModel = advancedOptions.fallbackModel;
				}
				if (advancedOptions.maxThinkingTokens && advancedOptions.maxThinkingTokens > 0) {
					queryOptions.options.maxThinkingTokens = advancedOptions.maxThinkingTokens;
				}

				// Tool permissions
				if (advancedOptions.allowedTools && advancedOptions.allowedTools.length > 0) {
					queryOptions.options.allowedTools = advancedOptions.allowedTools;
				}
				if (advancedOptions.disallowedTools && advancedOptions.disallowedTools.length > 0) {
					queryOptions.options.disallowedTools = advancedOptions.disallowedTools;
				}
				if (advancedOptions.additionalDirectories?.directories) {
					const dirs = advancedOptions.additionalDirectories.directories
						.map((d) => d.path?.trim())
						.filter((d) => d && d.length > 0);
					if (dirs.length > 0) {
						queryOptions.options.additionalDirectories = dirs;
					}
				}

				// Environment variables
				if (advancedOptions.environmentVariables?.variables) {
					const envVars: Record<string, string> = {};
					for (const v of advancedOptions.environmentVariables.variables) {
						if (v.name?.trim()) {
							envVars[v.name.trim()] = v.value || '';
						}
					}
					if (Object.keys(envVars).length > 0) {
						queryOptions.options.env = envVars;
					}
				}

				// Session persistence
				if (advancedOptions.persistSession === false) {
					queryOptions.options.persistSession = false;
				}

				// Settings sources
				if (advancedOptions.settingSources && advancedOptions.settingSources.length > 0) {
					queryOptions.options.settingSources = advancedOptions.settingSources;
				}

				// MCP Servers
				const mcpServersConfig = this.getNodeParameter('mcpServers', itemIndex, {}) as {
					servers?: Array<{
						name: string;
						transportType: string;
						command?: string;
						args?: string;
						env?: string;
						url?: string;
						headers?: string;
					}>;
				};

				if (mcpServersConfig.servers && mcpServersConfig.servers.length > 0) {
					const mcpServers: Record<string, any> = {};

					for (const server of mcpServersConfig.servers) {
						if (!server.name?.trim()) continue;
						const name = server.name.trim();

						switch (server.transportType) {
							case 'stdio': {
								if (!server.command?.trim()) {
									throw new NodeOperationError(
										this.getNode(),
										`MCP server "${name}": command is required for stdio transport`,
										{ itemIndex },
									);
								}
								const stdioConfig: any = {
									type: 'stdio',
									command: server.command.trim(),
								};
								if (server.args?.trim()) {
									stdioConfig.args = server.args.trim().split(/\s+/);
								}
								if (server.env?.trim() && server.env.trim() !== '{}') {
									try {
										stdioConfig.env = JSON.parse(server.env);
									} catch {
										throw new NodeOperationError(
											this.getNode(),
											`MCP server "${name}": invalid JSON in env`,
											{ itemIndex },
										);
									}
								}
								mcpServers[name] = stdioConfig;
								break;
							}
							case 'sse':
							case 'http': {
								if (!server.url?.trim()) {
									throw new NodeOperationError(
										this.getNode(),
										`MCP server "${name}": url is required for ${server.transportType} transport`,
										{ itemIndex },
									);
								}
								const httpConfig: any = {
									type: server.transportType,
									url: server.url.trim(),
								};
								if (
									server.headers?.trim() &&
									server.headers.trim() !== '{}'
								) {
									try {
										httpConfig.headers = JSON.parse(server.headers);
									} catch {
										throw new NodeOperationError(
											this.getNode(),
											`MCP server "${name}": invalid JSON in headers`,
											{ itemIndex },
										);
									}
								}
								mcpServers[name] = httpConfig;
								break;
							}
						}
					}

					if (Object.keys(mcpServers).length > 0) {
						queryOptions.options.mcpServers = {
							...queryOptions.options.mcpServers,
							...mcpServers,
						};
					}
				}

				// Custom Tools (inline)
				const customToolsConfig = this.getNodeParameter('customTools', itemIndex, {}) as {
					tools?: Array<{
						name: string;
						description: string;
						inputSchema: string;
						staticResponse: string;
					}>;
				};

				const allToolDefs = customToolsConfig.tools || [];

				const sdkTools: Array<ReturnType<typeof tool>> = [];

				if (allToolDefs.length > 0) {
					for (const t of allToolDefs) {
						if (!t.name?.trim()) continue;

						let schema: any = {};
						if (t.inputSchema?.trim()) {
							try {
								schema = JSON.parse(t.inputSchema);
							} catch {
								throw new NodeOperationError(
									this.getNode(),
									`Custom tool "${t.name}": invalid JSON in input schema`,
									{ itemIndex },
								);
							}
						}

						const zodShape = jsonSchemaToZodShape(schema);
						const response = t.staticResponse || '';

						sdkTools.push(
							tool(t.name.trim(), t.description || '', zodShape, async () => ({
								content: [{ type: 'text' as const, text: response }],
							})),
						);
					}
				}

				// Read connected AI tools (ai_tool input)
				try {
					const aiToolData = await this.getInputConnectionData(
						NodeConnectionTypes.AiTool,
						itemIndex,
					) as any[];
					if (Array.isArray(aiToolData)) {
						const lcTools = aiToolData.flatMap((t: any) =>
							Array.isArray(t?.tools) ? t.tools : [t],
						).filter((t: any) => t?.name);

						for (const lcTool of lcTools) {
							sdkTools.push(
								tool(
									lcTool.name,
									lcTool.description || '',
									lcTool.schema?.shape || {},
									async (input: any) => {
										try {
											const result = await lcTool.invoke(input);
											const text = typeof result === 'string'
												? result : JSON.stringify(result);
											return { content: [{ type: 'text' as const, text }] };
										} catch (error: any) {
											return { content: [{
												type: 'text' as const,
												text: `Error: ${error?.message || String(error)}`,
											}] };
										}
									},
								),
							);
						}
					}
				} catch { /* ai_tool input not connected */ }

				if (sdkTools.length > 0) {
					const customMcpServer = createSdkMcpServer({
						name: 'n8n-custom-tools',
						tools: sdkTools,
					});
					if (!queryOptions.options.mcpServers) {
						queryOptions.options.mcpServers = {};
					}
					queryOptions.options.mcpServers['n8n-custom-tools'] = customMcpServer;
				}

				// Streaming events
				if (advancedOptions.includePartialMessages) {
					queryOptions.options.includePartialMessages = true;
				}

				// ========== Execute Query ==========
				const messages: SDKMessage[] = [];
				const startTime = Date.now();

				if (advancedOptions.debug) {
					this.logger.info('Claude Agent session starting', {
						operation,
						model,
						permissionMode,
						itemIndex,
					});
				}

				for await (const message of query(queryOptions)) {
					// Filter streaming events unless explicitly requested
					if (!advancedOptions.includePartialMessages && message.type === 'stream_event') {
						continue;
					}

					messages.push(message);

					if (advancedOptions.debug) {
						this.logger.debug(`Message: ${message.type}`, {
							type: message.type,
							subtype: (message as any).subtype,
						});
					}
				}

				// ========== Format Output ==========
				let outputData: any;

				switch (outputFormat) {
					case 'text':
						outputData = formatAsText(messages);
						break;

					case 'summary':
						outputData = formatAsSummary(messages);
						break;

					case 'full':
						outputData = formatAsFull(messages);
						break;

					case 'structured':
						outputData = formatAsStructured(messages);
						break;
				}

				if (advancedOptions.debug) {
					const duration = Date.now() - startTime;
					this.logger.info('Claude Agent session completed', {
						duration_ms: duration,
						message_count: messages.length,
						success: outputData.success !== false,
					});
				}

				returnData.push({
					json: outputData,
					pairedItem: { item: itemIndex },
				});
			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({
						json: {
							error: error instanceof Error ? error.message : String(error),
							itemIndex,
						},
						pairedItem: itemIndex,
					});
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
}
