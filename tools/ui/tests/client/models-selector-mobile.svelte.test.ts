// Guards the phone model picker: the selector serves a phone a bottom drawer
// with the same search and sections the desktop menu shows.

import ModelsSelector from '$lib/components/app/models/ModelsSelector/ModelsSelector.svelte';
import { MODEL_ROW_WINDOW, SETTINGS_KEYS } from '$lib/constants';
import { ServerRole } from '$lib/enums';
import { ModelsService } from '$lib/services/models.service';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { ModelOption } from '$lib/types/models';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };
const option: ModelOption = {
	capabilities: [],
	id: 'org/Qwen3-8B:Q4_K_M',
	model: 'org/Qwen3-8B:Q4_K_M',
	name: 'Qwen3-8B'
};
const other: ModelOption = {
	capabilities: [],
	id: 'org/Qwen3-8B:Q8_0',
	model: 'org/Qwen3-8B:Q8_0',
	name: 'Qwen3-8B-Q8_0'
};

function catalog(count: number): ModelOption[] {
	return Array.from({ length: count }, (_, index) => {
		const model = `org/row-${index}:Q4_K_M`;

		return { capabilities: [], id: model, model, name: model };
	});
}

describe('model picker drawer on a phone', () => {
	let groupByFamily: boolean | undefined;

	beforeEach(async () => {
		serverStore.role = ServerRole.ROUTER;
		// the picker mounts a fetch of its own: serve it the same two models, so
		// the list it rebuilds does not clear the seeded one mid-test
		vi.spyOn(ModelsService, 'list').mockResolvedValue({
			data: [{ id: option.model }, { id: other.model }]
		} as never);
		modelsStore.models = [option, other];
		await modelsStore.selectModelById(option.id);
		modelsStore.setGroupOpen('local', true);
		groupByFamily = settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY];
		settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY] = false;
		await page.viewport(PHONE.width, PHONE.height);
	});

	afterEach(async () => {
		modelsStore.models = [];
		settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY] = groupByFamily;
		vi.restoreAllMocks();
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('opens the picker in a drawer', async () => {
		const screen = render(ModelsSelector);

		await screen.getByRole('button').click();

		await expect.element(screen.getByPlaceholder('Search models...')).toBeVisible();
		await expect.element(screen.getByText('Local models')).toBeVisible();
	});

	it('keeps keyboard selection inside the mounted row window', async () => {
		const models = catalog(MODEL_ROW_WINDOW + 1);
		const onModelChange = vi.fn();

		modelsStore.models = models;
		await modelsStore.selectModelById(models[0].id);
		vi.spyOn(ModelsService, 'list').mockResolvedValue({
			data: models.map((model) => ({ id: model.id }))
		} as never);

		const screen = render(ModelsSelector, { onModelChange });

		await screen.getByRole('button').click();
		expect(screen.container.querySelectorAll('[role="option"]')).toHaveLength(MODEL_ROW_WINDOW);

		await screen.getByPlaceholder('Search models...').click();
		await userEvent.keyboard('{ArrowDown}'.repeat(MODEL_ROW_WINDOW + 1));
		await userEvent.keyboard('{Enter}');

		expect(onModelChange).toHaveBeenCalledWith(models[0].id, models[0].model);
	});

	it('does not select a model from a collapsed local section', async () => {
		const onModelChange = vi.fn();
		const screen = render(ModelsSelector, { onModelChange });

		await screen.getByRole('button').click();
		await screen.getByRole('button', { name: 'Local models' }).click();
		await screen.getByPlaceholder('Search models...').click();
		await userEvent.keyboard('{ArrowDown}{Enter}');

		expect(onModelChange).not.toHaveBeenCalled();
	});
});
