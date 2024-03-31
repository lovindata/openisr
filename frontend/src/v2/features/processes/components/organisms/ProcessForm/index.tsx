import { ConfigurationContents } from "@/v2/features/processes/components/organisms/ProcessForm/ConfigurationContents";
import { ErrorContents } from "@/v2/features/processes/components/organisms/ProcessForm/ErrorContents";
import { BorderBox } from "@/v2/features/shared/components/atoms/BorderBox";
import { Header } from "@/v2/features/shared/components/atoms/Header";
import { components } from "@/v2/services/backend/endpoints";
import { Tab } from "@headlessui/react";

interface Props {
  card: components["schemas"]["CardMod"];
  onSuccessSubmit?: () => void;
}

export function ProcessForm({ card, onSuccessSubmit }: Props) {
  return (
    <BorderBox className="w-72 space-y-3 bg-black p-4">
      <Tab.Group>
        <Tab.List className="space-x-3">
          {card.status.type === "Errored" && (
            <>
              <Tab className="outline-none">
                {({ selected }) => (
                  <Header
                    name="Error"
                    className={selected ? "select-none" : "opacity-50"}
                  />
                )}
              </Tab>
              <span className="border-x" />
            </>
          )}
          <Tab className="outline-none">
            {({ selected }) => (
              <Header
                name="Configurations"
                className={selected ? "select-none" : "opacity-50"}
              />
            )}
          </Tab>
        </Tab.List>
        <Tab.Panels>
          {card.status.type === "Errored" && (
            <Tab.Panel>
              <ErrorContents
                error={card.status.error}
                imageId={card.image_id}
                onSuccessSubmit={onSuccessSubmit}
              />
            </Tab.Panel>
          )}
          <Tab.Panel>
            <ConfigurationContents
              card={card}
              onSuccessSubmit={onSuccessSubmit}
            />
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </BorderBox>
  );
}
