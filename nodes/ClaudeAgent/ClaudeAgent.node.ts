import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';
import { NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';
import {
	query,
	type SDKMessage,
	type SDKResultMessage,
	type SDKSystemMessage,
	type SDKAssistantMessage,
	type SDKUserMessage,
	type NonNullableUsage,
	type ModelUsage,
} from '@anthropic-ai/claude-agent-sdk';

// ========== 输出格式类型定义 ==========
interface TextOutput {
	result: string;
}

interface SummaryOutput {
	session_id: string;
	success: boolean;
	result: string | null;
	error_type?: 'error_max_turns' | 'error_during_execution';

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

// ========== 辅助函数 ==========
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

function formatAsText(messages: SDKMessage[]): TextOutput {
	const resultMsg = messages.find((m) => m.type === 'result') as SDKResultMessage | undefined;

	// 成功情况
	if (resultMsg?.subtype === 'success' && resultMsg.result) {
		return { result: resultMsg.result };
	}

	// 错误情况
	if (resultMsg?.subtype === 'error_max_turns') {
		return { result: 'Error: Maximum turns reached. Increase maxTurns or set to 0.' };
	}

	if (resultMsg?.subtype === 'error_during_execution') {
		return { result: 'Error: Execution failed. Enable debug mode for details.' };
	}

	// 尝试提取最后的助手消息
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

	// 统计工具使用
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

	// 统计消息
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

function formatAsFull(messages: SDKMessage[]): FullOutput {
	const systemInit = messages.find((m) => m.type === 'system' && (m as any).subtype === 'init') as
		| SDKSystemMessage
		| undefined;

	const resultMsg = messages.find((m) => m.type === 'result') as SDKResultMessage | undefined;

	// 构建时间线
	const timeline: FullOutput['parsed']['timeline'] = [];
	let currentTurn: any = null;

	messages.forEach((m) => {
		if (m.type === 'user' && !(m as any).isSynthetic) {
			// 新的用户消息 = 新一轮
			currentTurn = {
				turn: timeline.length + 1,
				user: extractUserText(m as SDKUserMessage),
				assistant: '',
				tools: [],
			};
			timeline.push(currentTurn);
		} else if (m.type === 'assistant' && currentTurn) {
			const assistantMsg = m as SDKAssistantMessage;

			// 提取文本
			const textContent = assistantMsg.message.content.find((c: any) => c.type === 'text');
			if (textContent) {
				currentTurn.assistant = (textContent as any).text;
			}

			// 提取工具使用
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

// ========== 节点类定义 ==========
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
		inputs: [{ type: NodeConnectionTypes.Main }],
		outputs: [{ type: NodeConnectionTypes.Main }],
		properties: [
			// ============ 会话管理 ============
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

			// ============ 提示词 ============
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

			// ============ 工作环境 ============
			{
				displayName: 'Working Directory',
				name: 'cwd',
				type: 'string',
				default: '',
				description: 'Directory where Claude Agent operates',
				placeholder: '/path/to/project',
				hint: 'Leave empty to use current directory',
			},

			// ============ 模型配置 ============
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
				],
				default: 'bypassPermissions',
				description: 'How to handle tool permissions',
			},

			// ============ 执行限制 ============
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

			// ============ 输出配置 ============
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
				],
				default: 'summary',
				description: 'How to format the output',
			},

			// ============ 高级选项 ============
			{
				displayName: 'Advanced Options',
				name: 'advancedOptions',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				options: [
					// --- 系统提示 ---
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

					// --- 模型配置 ---
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

					// --- 工具权限 ---
					{
						displayName: 'Allowed Tools',
						name: 'allowedTools',
						type: 'multiOptions',
						options: [
							{ name: 'Bash', value: 'Bash' },
							{ name: 'Edit', value: 'Edit' },
							{ name: 'Glob', value: 'Glob' },
							{ name: 'Grep', value: 'Grep' },
							{ name: 'Read', value: 'Read' },
							{ name: 'Task', value: 'Task' },
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
							{ name: 'Bash', value: 'Bash' },
							{ name: 'Edit', value: 'Edit' },
							{ name: 'Glob', value: 'Glob' },
							{ name: 'Grep', value: 'Grep' },
							{ name: 'Read', value: 'Read' },
							{ name: 'Task', value: 'Task' },
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

					// --- 调试 ---
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
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
			try {
				// ========== 获取参数 ==========
				const operation = this.getNodeParameter('operation', itemIndex) as string;
				const prompt = this.getNodeParameter('prompt', itemIndex) as string;
				const cwd = this.getNodeParameter('cwd', itemIndex, '') as string;
				const model = this.getNodeParameter('model', itemIndex) as string;
				const permissionMode = this.getNodeParameter('permissionMode', itemIndex) as string;
				const maxTurns = this.getNodeParameter('maxTurns', itemIndex) as number;
				const timeout = this.getNodeParameter('timeout', itemIndex) as number;
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
					debug?: boolean;
					includePartialMessages?: boolean;
				};

				// ========== 验证 ==========
				if (!prompt?.trim()) {
					throw new NodeOperationError(this.getNode(), 'Prompt is required', { itemIndex });
				}

				// ========== 构建 query 选项 ==========
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

				// 工作目录
				if (cwd?.trim()) {
					queryOptions.options.cwd = cwd.trim();
				}

				// 会话管理
				switch (operation) {
					case 'continue':
						queryOptions.options.continue = true;
						break;

					case 'resume':
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

					case 'fork':
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

				// 限制设置
				if (maxTurns > 0) {
					queryOptions.options.maxTurns = maxTurns;
				}

				if (timeout > 0) {
					const abortController = new AbortController();
					setTimeout(() => abortController.abort(), timeout * 1000);
					queryOptions.options.abortController = abortController;
				}

				// System prompt
				if (advancedOptions.customSystemPrompt?.trim()) {
					queryOptions.options.systemPrompt = advancedOptions.customSystemPrompt.trim();
				}

				// 模型配置
				if (advancedOptions.fallbackModel) {
					queryOptions.options.fallbackModel = advancedOptions.fallbackModel;
				}
				if (advancedOptions.maxThinkingTokens && advancedOptions.maxThinkingTokens > 0) {
					queryOptions.options.maxThinkingTokens = advancedOptions.maxThinkingTokens;
				}

				// 工具权限
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

				// 流式事件
				if (advancedOptions.includePartialMessages) {
					queryOptions.options.includePartialMessages = true;
				}

				// ========== 执行查询 ==========
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
					// 过滤流式事件（除非明确要求）
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

				// ========== 格式化输出 ==========
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
