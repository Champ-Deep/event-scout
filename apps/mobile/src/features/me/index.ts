import { outboxQueries } from '../../db/queries/outbox';

export const meFeature = {
  getOutboxStatus: async () => {
    const count = await outboxQueries.count();
    const pending = await outboxQueries.getPending(50);
    return { count, pending };
  },
};
